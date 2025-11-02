"""
Flask backend API for AusDevs conversation visualization.

Pure key-value server: serves pre-computed chunks data without any computation.
All clustering and coloring is pre-computed by compute_data.py and stored in cache.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_compress import Compress
import sqlite3
import json
import hashlib
from pathlib import Path
import os

# Get the path to the frontend dist folder (built React app)
DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"

app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path="")
CORS(app)
Compress(app)  # Enable gzip compression for all responses

DB_PATH = Path(__file__).parent / "conversation_data.db"


def init_cache_table():
    """Create cache table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT UNIQUE,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_db_connection():
    """Create database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/ausdevs_conversations/api/filters", methods=["GET"])
def get_filters():
    """Get all filter options (metadata from chunks table)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Vector types
        cursor.execute("SELECT DISTINCT vector_type FROM chunks ORDER BY vector_type")
        vector_types = [row[0] for row in cursor.fetchall()]

        # Methods
        cursor.execute("SELECT DISTINCT method FROM chunks ORDER BY method")
        methods = [row[0] for row in cursor.fetchall()]

        # Channels
        cursor.execute("SELECT DISTINCT channel FROM chunks ORDER BY channel")
        channels = ["All Channels"] + [row[0] for row in cursor.fetchall()]

        # Authors (top 200)
        cursor.execute("""
            SELECT author, COUNT(*) as count
            FROM messages
            GROUP BY author
            ORDER BY count DESC
            LIMIT 200
        """)
        authors = ["All Authors"] + [row[0] for row in cursor.fetchall()]

        # Clusters with sizes, sorted by size descending
        cursor.execute("""
            SELECT cluster, COUNT(*) as size
            FROM chunks
            WHERE vector_type = 'combined' AND method = 'pacmap'
            GROUP BY cluster
            ORDER BY size DESC
        """)
        cluster_data = cursor.fetchall()
        clusters = ["All Clusters"] + [f"Cluster {cluster} - {size} chunks" for cluster, size in cluster_data]

        conn.close()

        return jsonify({
            "vector_types": vector_types,
            "methods": methods,
            "channels": channels,
            "authors": authors,
            "clusters": clusters
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ausdevs_conversations/api/chunks", methods=["GET"])
def get_chunks():
    """Get pre-computed chunks data from cache (pure key-value lookup, no computation)."""
    try:
        # Get query parameters
        vector_type = request.args.get("vector_type", "combined")
        method = request.args.get("method", "umap")

        # Build cache key
        cache_key = f"get_chunks_{vector_type}_{method}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Look up pre-computed result in cache table
        cursor.execute(
            "SELECT result FROM cache WHERE cache_key = ?",
            (cache_key_hash,)
        )
        cached_result = cursor.fetchone()
        conn.close()

        if not cached_result:
            return jsonify({
                "error": f"No pre-computed cache for {vector_type}+{method}. Run compute_data.py first."
            }), 404

        # Return pre-computed result
        result_str = cached_result[0]
        result_data = json.loads(result_str)
        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ausdevs_conversations/api/chunk/<int:chunk_id>", methods=["GET"])
def get_chunk(chunk_id):
    """Get full conversation messages and descriptions for a chunk."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get chunk info including full descriptions
        cursor.execute(
            "SELECT channel, chunk_index, chunk_start, chunk_end, msg_count, topic, technical_topic, sentiment FROM chunks WHERE id = ? LIMIT 1",
            (chunk_id,)
        )
        chunk_info = cursor.fetchone()

        if not chunk_info:
            return jsonify({"error": "Chunk not found"}), 404

        channel, chunk_index, start, end, msg_count, topic, technical_topic, sentiment = chunk_info

        # Find the 'topic' + 'umap' chunk record for this (channel, chunk_index)
        # to ensure we get messages (messages are only stored for topic+umap records)
        cursor.execute(
            "SELECT id FROM chunks WHERE channel = ? AND chunk_index = ? AND vector_type = 'topic' AND method = 'umap' LIMIT 1",
            (channel, chunk_index)
        )
        messages_chunk_result = cursor.fetchone()

        if messages_chunk_result:
            messages_chunk_id = messages_chunk_result[0]
            # Get messages from the topic+umap record
            cursor.execute(
                "SELECT author, content, timestamp FROM messages WHERE chunk_id = ? ORDER BY id",
                (messages_chunk_id,)
            )
            messages = cursor.fetchall()
        else:
            # Fallback: try to get messages from the chunk_id directly
            cursor.execute(
                "SELECT author, content, timestamp FROM messages WHERE chunk_id = ? ORDER BY id",
                (chunk_id,)
            )
            messages = cursor.fetchall()

        conn.close()

        # Format messages
        formatted_messages = [
            {
                'author': msg[0],
                'content': msg[1],
                'timestamp': msg[2]
            }
            for msg in messages
        ]

        return jsonify({
            'chunk_id': chunk_id,
            'channel': channel,
            'start': start,
            'end': end,
            'message_count': msg_count,
            'topic': topic or "N/A",
            'technical_topic': technical_topic or "N/A",
            'sentiment': sentiment or "N/A",
            'messages': formatted_messages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/ausdevs_conversations/", defaults={"path": ""})
@app.route("/ausdevs_conversations/<path:path>")
def serve_frontend(path):
    """Serve the React frontend, with fallback to index.html for client-side routing."""
    # If it's an API route, let it be handled by the API endpoints above
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404

    # Try to serve the file if it exists
    if path and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)

    # Fallback to index.html for client-side routing
    index_path = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(DIST_DIR, "index.html")

    return jsonify({"error": "Frontend not found"}), 404


if __name__ == "__main__":
    print(f"Starting server with database at: {DB_PATH}")
    print("Initializing cache table...")
    init_cache_table()
    print("Cache table ready!")
    print("Note: Run 'python compute_data.py' to pre-compute cache entries.")
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=False, port=9000, host="0.0.0.0")
