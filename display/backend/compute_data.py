"""
Compute cache data for all vector_type + method combinations.

This script:
1. Loads raw embeddings from JSON files (768 or 2304 dimensions)
2. Clusters on raw high-dimensional vectors ONCE per vector_type using MiniBatchKMeans
3. Stores cluster IDs in database (shared across all methods)
4. Pre-computes cache for all 12 combinations (4 vector types × 3 methods)

Backend.py simply looks up these pre-computed results (pure key-value server).

Usage:
    python compute_data.py
"""

import sqlite3
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from cuml.cluster import KMeans as GPUKMeans
import cupy as cp

DB_PATH = Path(__file__).parent / "conversation_data.db"
# Raw embeddings are stored in the aus_devs/output folder
# From backend/compute_data.py: go up to display, then to aus_devs, then output
EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "output"


def get_db_connection():
    """Create database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_cache_table():
    """Create cache table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY,
            cache_key TEXT UNIQUE,
            result TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def load_all_embeddings(vector_type):
    """
    Load all embeddings from JSON files for a given vector_type.
    Returns:
        embeddings_array: numpy array of shape (n_chunks, embedding_dim)
        chunk_ids: list of chunk IDs in same order as embeddings_array
        chunk_info: dict mapping chunk_id -> (channel, chunk_index, x, y, metadata)
    """
    print(f"  Loading embeddings for {vector_type}...")

    all_embeddings = []
    chunk_ids = []
    chunk_info = {}

    # Get all unique channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT channel FROM chunks ORDER BY channel")
    channels = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Load embeddings for each channel
    for channel in channels:
        # Find matching JSON file by looking for files that start with channel name
        # (glob doesn't work well with square brackets in filenames)
        json_files = [f for f in EMBEDDINGS_DIR.glob("*_embeddings.json")
                      if f.name.startswith(channel)]
        if not json_files:
            print(f"    Warning: No JSON found for {channel}")
            continue

        json_file = json_files[0]
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"    Warning: Could not load {json_file}: {e}")
            continue

        # Get chunk info from database for this channel
        # IMPORTANT: Only load from pacmap method to avoid loading same chunk 3x (one per method)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, chunk_index, x, y, topic_short, technical_topic_short, sentiment_short FROM chunks "
            "WHERE channel = ? AND vector_type = ? AND method = 'pacmap' ORDER BY chunk_index",
            (channel, vector_type)
        )
        db_chunks = cursor.fetchall()
        conn.close()

        # Map JSON chunks (by index) to database chunks
        db_by_idx = {row[1]: row for row in db_chunks}  # chunk_index -> row

        # Load embeddings matching database order
        for chunk_idx, json_chunk in enumerate(json_data):
            if chunk_idx not in db_by_idx:
                continue

            db_row = db_by_idx[chunk_idx]
            chunk_id = db_row[0]

            # Get the embedding for this vector_type
            embedding_key = {
                'topic': 'topic_embedding',
                'technical_topic': 'technical_topic_embedding',
                'sentiment': 'sentiment_embedding',
                'combined': 'combined_embedding',
            }[vector_type]

            embedding = json_chunk.get(embedding_key)
            if embedding and len(embedding) > 0:
                all_embeddings.append(np.array(embedding, dtype=np.float32))
                chunk_ids.append(chunk_id)
                chunk_info[chunk_id] = {
                    'channel': channel,
                    'chunk_index': chunk_idx,
                    'x': db_row[2],
                    'y': db_row[3],
                    'topic_short': db_row[4],
                    'technical_topic_short': db_row[5],
                    'sentiment_short': db_row[6],
                }

    if not all_embeddings:
        print(f"  ERROR: No embeddings loaded for {vector_type}")
        return None, [], {}

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"  Loaded {len(embeddings_array)} embeddings ({embeddings_array.shape[1]} dims)")

    return embeddings_array, chunk_ids, chunk_info


def cluster_with_minibatch_kmeans(embeddings_array, n_clusters=300):
    """
    Cluster embeddings using GPU-accelerated KMeans via cuML.
    """
    print(f"  Clustering with GPU KMeans (k={n_clusters})...")

    # Convert numpy array to cupy (GPU) array
    gpu_embeddings = cp.asarray(embeddings_array)
    print(f"    Transferred {embeddings_array.shape[0]} embeddings to GPU")

    # Use cuML's GPU-accelerated KMeans
    kmeans = GPUKMeans(
        n_clusters=n_clusters,
        random_state=42,
        verbose=1,
        init='kmeans++',
        n_init=1  # Single init on GPU (much faster than CPU)
    )

    # Fit on GPU
    kmeans.fit(gpu_embeddings)

    # Get labels and convert back to numpy
    labels = cp.asnumpy(kmeans.labels_)
    print(f"  Clustering complete!")

    # Verify all points assigned
    unique_labels = np.unique(labels)
    print(f"  Found {len(unique_labels)} clusters")
    if -1 in unique_labels:
        print(f"  WARNING: Found {(labels == -1).sum()} noise points, reassigning...")
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        noise_points = gpu_embeddings[noise_indices]
        # Get cluster centers from the model
        cluster_centers = kmeans.cluster_centers_
        distances = cp.linalg.norm(noise_points[:, cp.newaxis, :] - cluster_centers, axis=2)
        nearest_clusters = cp.argmin(distances, axis=1)
        labels[noise_indices] = cp.asnumpy(nearest_clusters)

    return labels


def store_clusters_in_db(vector_type, chunk_ids, labels):
    """
    Store cluster assignments in database.
    """
    print(f"  Storing clusters in database...")
    conn = get_db_connection()
    cursor = conn.cursor()

    for chunk_id, cluster_id in zip(chunk_ids, labels):
        cursor.execute(
            "UPDATE chunks SET cluster = ? WHERE id = ? AND vector_type = ?",
            (int(cluster_id), chunk_id, vector_type)
        )

    conn.commit()
    conn.close()


def copy_clusters_across_methods(vector_type):
    """
    Copy cluster IDs from pacmap method to pca and umap methods for a given vector_type.
    This ensures all methods share the same cluster assignments from the raw embedding clustering.
    Uses ID-based batched updates for performance (avoiding slow text-based WHERE clauses).
    """
    print(f"  Copying clusters across methods for {vector_type}...", end=" ", flush=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Load pacmap clusters: map (channel, chunk_index) -> cluster_id
    cursor.execute(
        "SELECT channel, chunk_index, cluster FROM chunks WHERE vector_type = ? AND method = 'pacmap'",
        (vector_type,)
    )
    clusters_map = {(channel, chunk_idx): cluster for channel, chunk_idx, cluster in cursor.fetchall()}

    # For each destination method, copy clusters using ID-based batched updates
    for dest_method in ['pca', 'umap']:
        # Load destination method row IDs
        cursor.execute(
            "SELECT id, channel, chunk_index FROM chunks WHERE vector_type = ? AND method = ?",
            (vector_type, dest_method)
        )
        dest_rows = cursor.fetchall()

        # Build update tuples: (cluster_id, row_id)
        updates = [
            (clusters_map.get((channel, chunk_idx)), row_id)
            for row_id, channel, chunk_idx in dest_rows
            if (channel, chunk_idx) in clusters_map
        ]

        # Execute batched updates in chunks of 500
        batch_size = 500
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i+batch_size]
            cursor.executemany(
                "UPDATE chunks SET cluster = ? WHERE id = ?",
                batch
            )
            conn.commit()

    conn.close()
    print("✓")


def compute_chunks_from_db(vector_type, method):
    """
    Build cache entry by reading pre-computed clusters from database.
    Applies the same clusters across all methods.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all chunks with pre-computed clusters for this vector_type
        cursor.execute(
            """
            SELECT id, channel, x, y, cluster, authors, msg_count,
                   chunk_start, chunk_end, topic, topic_short,
                   technical_topic, technical_topic_short,
                   sentiment, sentiment_short
            FROM chunks
            WHERE vector_type = ? AND method = ?
            ORDER BY id
            """,
            (vector_type, method)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"chunks": [], "total": 0}

        # Generate distinct colors for 300 clusters using HSL color space
        def hsl_to_hex(h, s, l):
            import colorsys
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

        # Create 300 evenly-spaced colors across the hue spectrum
        colors = [hsl_to_hex(i / 300.0, 0.6, 0.5) for i in range(300)]

        chunks_data = []
        for row in rows:
            chunk_id, channel_name, x, y, cluster_id, authors_json, msg_count, \
                start, end, topic, topic_s, tech, tech_s, sent, sent_s = row

            authors_list = json.loads(authors_json)

            if cluster_id is None:
                cluster_id = 0

            # Clamp cluster_id to valid range
            cluster_id = min(cluster_id, 299)
            cluster_color = colors[cluster_id]
            # Opacity based on cluster size (smaller clusters = more transparent to see outliers)
            opacity = 0.5

            chunks_data.append({
                'id': chunk_id,
                'x': x,
                'y': y,
                'channel': channel_name,
                'topic': topic_s or "N/A",
                'technical_topic': tech_s or "N/A",
                'sentiment': sent_s or "N/A",
                'messages': msg_count,
                'authors': ', '.join(authors_list),
                'color': cluster_color,
                'opacity': opacity,
                'cluster': cluster_id,
                'start': start,
                'end': end
            })

        return {"chunks": chunks_data, "total": len(chunks_data)}

    except Exception as e:
        print(f"ERROR computing {vector_type}+{method}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def cache_result(cache_key, result_data):
    """Store result in cache table."""
    cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    result_str = json.dumps(result_data)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO cache (cache_key, result)
        VALUES (?, ?)
        """,
        (cache_key_hash, result_str)
    )
    conn.commit()
    conn.close()


def main():
    """Main: cluster and cache all vector_type + method combinations."""
    print("\n" + "=" * 60)
    print("AusDevs High-Dimensional Clustering")
    print("=" * 60)

    # Initialize cache table
    init_cache_table()

    # Get all unique vector types and methods
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT vector_type FROM chunks ORDER BY vector_type")
    vector_types = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT method FROM chunks ORDER BY method")
    methods = [row[0] for row in cursor.fetchall()]

    conn.close()

    total_combinations = len(vector_types) * len(methods)
    print(f"\nVector types: {vector_types}")
    print(f"Methods: {methods}")
    print(f"Total combinations: {total_combinations}\n")

    # Clear old cache
    print("Clearing old cache...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache")
    conn.commit()
    conn.close()
    print("✓ Cache cleared\n")

    # Phase 1: Cluster each vector_type on raw embeddings (do once)
    print("=" * 60)
    print("PHASE 1: Clustering on raw embeddings")
    print("=" * 60 + "\n")

    for vector_type in vector_types:
        print(f"[{vector_type}] Clustering...")

        # Load embeddings
        embeddings_array, chunk_ids, chunk_info = load_all_embeddings(vector_type)
        if embeddings_array is None:
            print(f"  ✗ Skipping {vector_type}\n")
            continue

        # Cluster
        labels = cluster_with_minibatch_kmeans(embeddings_array, n_clusters=300)

        # Store in database
        store_clusters_in_db(vector_type, chunk_ids, labels)
        print(f"  ✓ {vector_type} clustered and stored\n")

        # Free memory
        del embeddings_array, chunk_ids, chunk_info, labels

    # Copy cluster IDs from pacmap to pca and umap for each vector_type
    print("\nCopying clusters across all methods...")
    for vector_type in vector_types:
        copy_clusters_across_methods(vector_type)
    print()

    # Phase 2: Build cache for all combinations (use pre-computed clusters)
    print("=" * 60)
    print("PHASE 2: Building cache for all methods")
    print("=" * 60 + "\n")

    count = 0
    errors = 0

    for vector_type in vector_types:
        for method in methods:
            count += 1
            print(f"[{count}/{total_combinations}] {vector_type} + {method}...", end=" ", flush=True)

            try:
                result = compute_chunks_from_db(vector_type, method)
                if "error" in result:
                    print(f"✗ {result['error']}")
                    errors += 1
                else:
                    cache_key = f"get_chunks_{vector_type}_{method}"
                    cache_result(cache_key, result)
                    chunk_count = result.get("total", 0)
                    print(f"✓ ({chunk_count} chunks)")
            except Exception as e:
                print(f"✗ {str(e)[:60]}")
                errors += 1

    print(f"\n" + "=" * 60)
    print(f"Computation complete!")
    print(f"  Computed: {count - errors}/{total_combinations}")
    print(f"  Errors: {errors}")
    print(f"=" * 60 + "\n")

    return errors == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
