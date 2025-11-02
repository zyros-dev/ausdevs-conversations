"""
Build SQLite database from embedding JSON files for fast queries.

Converts the output/*.json files into a lightweight SQLite database,
making it suitable for Gradio app with on-demand data loading.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import umap
from sklearn.decomposition import PCA

try:
    import pacmap
    has_pacmap = True
except ImportError:
    has_pacmap = False
    print("Note: pacmap not installed. Skipping PACMAP in database.")


def load_embeddings(output_dir: str = "output") -> Dict[str, List[dict]]:
    """Load all embedding JSON files from output directory."""
    embeddings_by_channel = {}
    output_path = Path(output_dir)

    for json_file in output_path.glob("*_embeddings.json"):
        channel_name = json_file.stem.replace("_embeddings", "")
        with open(json_file, 'r') as f:
            embeddings_by_channel[channel_name] = json.load(f)
            print(f"Loaded {channel_name}: {len(embeddings_by_channel[channel_name])} chunks")

    return embeddings_by_channel


def build_database(embeddings_by_channel: Dict, db_path: str = "conversation_data.db"):
    """
    Build SQLite database from embeddings.

    Creates two tables:
    - chunks: Metadata for all chunks (coordinates, short descriptions, etc.)
    - messages: Full message content (loaded on-demand when chunk clicked)
    """
    # Remove old database if exists
    Path(db_path).unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            vector_type TEXT NOT NULL,
            method TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            cluster INTEGER NOT NULL,
            authors TEXT NOT NULL,
            msg_count INTEGER NOT NULL,
            chunk_start TEXT NOT NULL,
            chunk_end TEXT NOT NULL,
            topic TEXT,
            topic_short TEXT,
            technical_topic TEXT,
            technical_topic_short TEXT,
            sentiment TEXT,
            sentiment_short TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
    ''')

    # Extract vector data for all reduction methods
    print("\n" + "="*80)
    print("Extracting vectors and building coordinates...")
    print("="*80)

    vector_types = ['topic', 'technical_topic', 'sentiment', 'combined']
    reduction_methods = ['umap', 'pca']
    if has_pacmap:
        reduction_methods.append('pacmap')

    # Collect all vectors by type
    all_vectors = {vt: [] for vt in vector_types}
    all_metadata = []
    chunk_id_map = {}  # (channel, chunk_index) -> list of chunk_ids

    total_filtered = 0
    chunk_counter = 0

    for channel_name, chunks in embeddings_by_channel.items():
        for i, chunk in enumerate(chunks):
            msg_count = len(chunk['messages'])

            # Store ALL chunks - let Gradio filter by min_messages
            # (don't skip any chunks here)

            # Collect vectors
            for vt in vector_types:
                all_vectors[vt].append(np.array(chunk[f'{vt}_embedding']))

            # Store metadata
            all_metadata.append({
                'channel': channel_name,
                'chunk_index': i,
                'messages': msg_count,
                'authors': list(set(msg['author'] for msg in chunk['messages'])),
                'chunk_start': chunk['chunk_start'],
                'chunk_end': chunk['chunk_end'],
                'topic': chunk.get('topic', ''),
                'topic_short': chunk.get('topic_short', ''),
                'technical_topic': chunk.get('technical_topic', ''),
                'technical_topic_short': chunk.get('technical_topic_short', ''),
                'sentiment': chunk.get('sentiment', ''),
                'sentiment_short': chunk.get('sentiment_short', ''),
                'messages_data': chunk['messages'],
            })

            chunk_id_map[(channel_name, i)] = chunk_counter
            chunk_counter += 1

    print(f"Processing {len(all_metadata)} chunks with {len(reduction_methods)} methods × {len(vector_types)} vector types")

    # Convert to numpy arrays
    for vt in vector_types:
        all_vectors[vt] = np.array(all_vectors[vt])

    # Compute 2D coordinates for each method
    coords_by_method = {}

    for method in reduction_methods:
        print(f"\nComputing {method.upper()} coordinates...")
        coords_by_method[method] = {}

        for vt in vector_types:
            print(f"  - {vt}...", end=" ", flush=True)

            if method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                coords = reducer.fit_transform(all_vectors[vt])
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(all_vectors[vt])
            elif method == 'pacmap':
                reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=42)
                coords = reducer.fit_transform(all_vectors[vt])

            coords_by_method[method][vt] = coords
            print("✓")

    # Insert into database
    print("\n" + "="*80)
    print("Inserting into database...")
    print("="*80)

    chunk_ids_in_db = {}  # Store IDs for message insertion

    for idx, meta in enumerate(all_metadata):
        channel = meta['channel']
        chunk_index = meta['chunk_index']

        for vt in vector_types:
            for method in reduction_methods:
                coords = coords_by_method[method][vt]
                x, y = float(coords[idx, 0]), float(coords[idx, 1])

                # Insert chunk record
                cursor.execute('''
                    INSERT INTO chunks
                    (channel, chunk_index, vector_type, method, x, y, cluster, authors,
                     msg_count, chunk_start, chunk_end, topic, topic_short, technical_topic, technical_topic_short, sentiment, sentiment_short)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    channel,
                    chunk_index,
                    vt,
                    method,
                    x, y,
                    -1,  # cluster placeholder (will be computed per query)
                    json.dumps(meta['authors']),
                    meta['messages'],
                    meta['chunk_start'],
                    meta['chunk_end'],
                    meta['topic'],
                    meta['topic_short'],
                    meta['technical_topic'],
                    meta['technical_topic_short'],
                    meta['sentiment'],
                    meta['sentiment_short'],
                ))

                # Store ID for first vector_type/method combo to avoid duplicating messages
                if vt == 'topic' and method == 'umap':
                    chunk_id = cursor.lastrowid
                    chunk_ids_in_db[(channel, chunk_index)] = chunk_id

    # Insert messages (only once per chunk)
    print("Inserting messages...")
    for idx, meta in enumerate(all_metadata):
        channel = meta['channel']
        chunk_index = meta['chunk_index']
        chunk_id = chunk_ids_in_db[(channel, chunk_index)]

        for msg in meta['messages_data']:
            cursor.execute('''
                INSERT INTO messages (chunk_id, author, content, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (chunk_id, msg['author'], msg['content'], msg['timestamp']))

    # Create indices for fast queries
    print("Creating indices...")
    cursor.execute('CREATE INDEX idx_vector_type ON chunks(vector_type)')
    cursor.execute('CREATE INDEX idx_method ON chunks(method)')
    cursor.execute('CREATE INDEX idx_channel ON chunks(channel)')
    cursor.execute('CREATE INDEX idx_msg_chunk ON messages(chunk_id)')

    conn.commit()
    conn.close()

    db_size_mb = Path(db_path).stat().st_size / (1024**2)
    print(f"\n✓ Database created: {db_path} ({db_size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Building SQLite database from embeddings...")
    embeddings = load_embeddings()
    build_database(embeddings)
    print("\nDone! Run 'python gradio_visualization.py' to start the app.")
