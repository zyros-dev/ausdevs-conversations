"""
Generate interactive HTML visualization of Discord conversation embeddings.

Creates a self-contained HTML file with:
- UMAP dimensionality reduction
- DBSCAN clustering
- Interactive dropdowns for vector selection and username filtering
- Plotly.js for interactive visualization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import base64

try:
    import umap
    from sklearn.cluster import DBSCAN
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install umap-learn scikit-learn plotly matplotlib")
    exit(1)


def get_cluster_colors() -> List[str]:
    """
    Get the 10 high-contrast colors from matplotlib's tab10 palette.
    """
    cmap = plt.get_cmap('tab10')
    colors = [f'#{int(cmap(i)[:3][0]*255):02x}{int(cmap(i)[:3][1]*255):02x}{int(cmap(i)[:3][2]*255):02x}'
              for i in range(10)]
    return colors


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


def extract_vector_data(embeddings_by_channel: Dict, min_messages: int = 5) -> Dict[str, dict]:
    """
    Extract all vectors and metadata.

    Args:
        embeddings_by_channel: Dict of channel embeddings
        min_messages: Minimum number of messages per chunk (default: 5, filters out chunks with <=4 messages)

    Returns dict with vector types as keys, each containing:
    - vectors: numpy array of embeddings
    - metadata: list of dicts with chunk info
    """
    vector_types = {
        'topic': [],
        'technical_topic': [],
        'sentiment': [],
        'combined': []
    }

    metadata = []
    filtered_count = 0

    for channel_name, chunks in embeddings_by_channel.items():
        for i, chunk in enumerate(chunks):
            msg_count = len(chunk['messages'])

            # Skip chunks with too few messages
            if msg_count < min_messages:
                filtered_count += 1
                continue

            # Extract vectors
            for vec_type in vector_types:
                vector_types[vec_type].append(chunk[f'{vec_type}_embedding'])

            # Extract metadata (only include short descriptions to save space)
            metadata.append({
                'channel': channel_name,
                'chunk_index': i,
                'messages': msg_count,
                'authors': list(set(msg['author'] for msg in chunk['messages'])),
                'chunk_start': chunk['chunk_start'],
                'chunk_end': chunk['chunk_end'],
                'topic_short': chunk.get('topic_short', ''),
                'technical_topic_short': chunk.get('technical_topic_short', ''),
                'sentiment_short': chunk.get('sentiment_short', ''),
            })

    print(f"Filtered out {filtered_count} chunks with < {min_messages} messages")

    # Convert to numpy arrays
    for vec_type in vector_types:
        vector_types[vec_type] = np.array(vector_types[vec_type])

    return {
        'vectors': vector_types,
        'metadata': metadata
    }


def reduce_and_cluster(vectors: Dict[str, np.ndarray], metadata: List[dict]) -> Dict[str, dict]:
    """
    Apply multiple dimensionality reduction methods and DBSCAN clustering to each vector type.

    Returns dict with processed data for each vector type and reduction method.
    """
    try:
        import pacmap
        has_pacmap = True
    except ImportError:
        has_pacmap = False
        print("Note: pacmap not installed. Install with: pip install pacmap")

    from sklearn.decomposition import PCA

    results = {}
    reduction_methods = ['umap', 'pca']
    if has_pacmap:
        reduction_methods.append('pacmap')

    for vec_type, embeddings in vectors.items():
        print(f"\nProcessing {vec_type} vectors ({embeddings.shape[0]} samples)...")
        results[vec_type] = {}

        for method in reduction_methods:
            print(f"  - Running {method.upper()}...")

            # Apply dimensionality reduction
            if method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                coords = reducer.fit_transform(embeddings)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(embeddings)
            elif method == 'pacmap':
                reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=42)
                coords = reducer.fit_transform(embeddings)

            # DBSCAN clustering on reduced coordinates
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(coords)

            # Split large clusters (max 100 points per cluster)
            labels = split_large_clusters(labels, max_size=100)

            # Identify noise points (label = -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"    Found {n_clusters} clusters, {n_noise} noise points")

            results[vec_type][method] = {
                'coords': coords,
                'cluster_labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
            }

    return results


def split_large_clusters(labels: np.ndarray, max_size: int = 100) -> np.ndarray:
    """
    Split clusters larger than max_size into subclusters.

    Args:
        labels: DBSCAN cluster labels
        max_size: Maximum cluster size before splitting

    Returns:
        Updated labels with large clusters split
    """
    new_labels = labels.copy()
    next_label = max(new_labels) + 1

    for cluster_id in range(max(labels) + 1):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > max_size:
            # Split into subclusters
            n_splits = (len(cluster_indices) + max_size - 1) // max_size
            split_size = len(cluster_indices) // n_splits

            for i in range(n_splits):
                start = i * split_size
                end = start + split_size if i < n_splits - 1 else len(cluster_indices)
                new_labels[cluster_indices[start:end]] = next_label
                next_label += 1

    return new_labels


def generate_html(data: dict, metadata: List[dict], embeddings_by_channel: Dict, output_file: str = "visualization.html"):
    """
    Generate self-contained HTML file with interactive visualization.

    Author dropdown limited to top 200 most active authors.
    Channel dropdown includes all channels.
    """
    # Prepare data for embedding in HTML
    vector_types = list(data.keys())
    reduction_methods = list(data[list(data.keys())[0]].keys())  # Get methods from first vector type

    # Get top 200 authors by message count
    author_counts = {}
    for meta in metadata:
        for author in meta['authors']:
            author_counts[author] = author_counts.get(author, 0) + meta['messages']

    # Sort by message count and keep top 200
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:200]
    all_authors = [author for author, _ in top_authors]

    # Get all unique channels
    all_channels = sorted(set(meta['channel'] for meta in metadata))

    # Create traces for each vector type and reduction method
    traces_data = {}

    for vec_type in vector_types:
        traces_data[vec_type] = {}

        for method in reduction_methods:
            coords = data[vec_type][method]['coords']
            labels = data[vec_type][method]['cluster_labels']

            # Store only the coordinates and basic info - let JavaScript filter by author/channel
            # This avoids 200x duplication of the same coordinate data
            points = []
            for i, meta in enumerate(metadata):
                points.append({
                    'x': float(coords[i, 0]),
                    'y': float(coords[i, 1]),
                    'cluster': int(labels[i]),
                    'authors': meta['authors'],
                    'channel': meta['channel'],
                    'metadata': meta,
                })
            traces_data[vec_type][method] = points

    # Get tab10 colors
    cluster_colors = get_cluster_colors()

    # Generate HTML
    html = generate_html_template(traces_data, vector_types, reduction_methods, all_authors, all_channels, cluster_colors, embeddings_by_channel)

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\n✓ Visualization saved to {output_file}")
    print(f"  Vector types: {', '.join(vector_types)}")
    print(f"  Authors: {len(all_authors)}")


def generate_html_template(traces_data: dict, vector_types: List[str], reduction_methods: List[str], all_authors: List[str], all_channels: List[str], cluster_colors: List[str], embeddings_by_channel: Dict = None) -> str:
    """Generate the complete HTML template with embedded data."""

    # JSON-encode the traces data
    traces_json = json.dumps(traces_data)
    colors_json = json.dumps(cluster_colors)

    # Create a compressed embeddings structure: just channel -> chunk_index -> messages
    # We only store messages for the hover detail view, not the vectors
    compressed_embeddings = {}
    if embeddings_by_channel:
        for channel_name, chunks in embeddings_by_channel.items():
            compressed_embeddings[channel_name] = {}
            for i, chunk in enumerate(chunks):
                compressed_embeddings[channel_name][str(i)] = chunk.get('messages', [])

    embeddings_json = json.dumps(compressed_embeddings)

    # Calculate size for debugging
    import sys
    embeddings_size_mb = sys.getsizeof(embeddings_json) / (1024**2)
    traces_size_mb = sys.getsizeof(traces_json) / (1024**2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discord Conversation Embedding Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 4px;
        }}

        .controls {{
            background: white;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .control-group label {{
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }}

        .control-group select {{
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            min-width: 200px;
        }}

        .control-group select:hover {{
            border-color: #999;
        }}

        .control-group select:focus {{
            outline: none;
            border-color: #666;
            box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.05);
        }}

        .plot-container {{
            width: 100%;
            height: 700px;
            background: white;
            margin: 20px 0;
        }}

        #plotDiv {{
            width: 100% !important;
            height: 100% !important;
        }}


        .tooltip {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            max-width: 300px;
            font-size: 0.85em;
            line-height: 1.5;
        }}

        .tooltip-title {{
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .tooltip-item {{
            margin: 5px 0;
        }}

        .tooltip-label {{
            font-weight: 500;
            color: #667eea;
        }}

        .chatbox {{
            background: white;
            border: 1px solid #e0e0e0;
            margin: 20px;
            max-height: 600px;
            overflow-y: auto;
            display: none;
        }}

        .chatbox.active {{
            display: block;
        }}

        .chatbox-header {{
            background: #f5f5f5;
            padding: 15px 20px;
            border-bottom: 1px solid #e0e0e0;
            font-weight: 600;
            color: #333;
        }}

        .messages-container {{
            padding: 20px;
        }}

        .message {{
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid #f0f0f0;
        }}

        .message:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}

        .message-author {{
            font-weight: 600;
            color: #667eea;
            font-size: 0.95em;
        }}

        .message-time {{
            font-size: 0.85em;
            color: #999;
            margin-top: 4px;
        }}

        .message-content {{
            margin-top: 8px;
            color: #333;
            line-height: 1.5;
            word-break: break-word;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div style="padding: 20px; border-bottom: 1px solid #e0e0e0;">
            <h1 style="font-size: 1.8em; margin: 0; color: #333;">AusDevs 2.0.0 Conversation Analysis</h1>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="vectorSelect">Vector Type:</label>
                <select id="vectorSelect">
                    {''.join(f'<option value="{vt}"{"" if vt != "combined" else " selected"}>{vt.replace("_", " ").title()}</option>' for vt in vector_types)}
                </select>
            </div>

            <div class="control-group">
                <label for="methodSelect">Dimensionality Reduction:</label>
                <select id="methodSelect">
                    {''.join(f'<option value="{m}"{"" if m != "umap" else " selected"}>{m.upper()}</option>' for m in reduction_methods)}
                </select>
            </div>

            <div class="control-group">
                <label for="channelSelect">Filter by Channel (optional):</label>
                <select id="channelSelect">
                    <option value="all">All Channels</option>
                    {''.join(f'<option value="{ch}">{ch}</option>' for ch in all_channels)}
                </select>
            </div>

            <div class="control-group">
                <label for="authorSelect">Filter by Author (optional):</label>
                <select id="authorSelect">
                    <option value="all">All Authors</option>
                    {''.join(f'<option value="{author}">{author}</option>' for author in all_authors)}
                </select>
            </div>
        </div>

        <div class="plot-container">
            <div id="plotDiv"></div>
        </div>

        <div class="chatbox" id="chatbox">
            <div class="chatbox-header">
                Conversation Messages
                <span style="float: right; cursor: pointer; font-size: 1.2em;" onclick="document.getElementById('chatbox').classList.remove('active')">×</span>
            </div>
            <div class="messages-container" id="messagesContainer">
            </div>
        </div>
    </div>

    <script>
        const tracesData = {traces_json};
        const vectorTypes = {json.dumps(vector_types)};
        const reductionMethods = {json.dumps(reduction_methods)};
        const allAuthors = {json.dumps(['all'] + all_authors)};
        const allChannels = {json.dumps(['all'] + all_channels)};
        const clusterColors = {colors_json};
        const embeddingsData = {embeddings_json};

        // Convert ISO timestamp to GMT+10 string
        function formatTimestamp(isoString) {{
            try {{
                const date = new Date(isoString);
                // Convert to GMT+10 (AEST/AEDT depending on daylight saving)
                const formatter = new Intl.DateTimeFormat('en-AU', {{
                    timeZone: 'Australia/Sydney',
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }});
                return formatter.format(date);
            }} catch (e) {{
                return isoString;
            }}
        }}

        // Display messages in chatbox
        function displayMessages(metadata) {{
            const container = document.getElementById('messagesContainer');
            container.innerHTML = '';

            const header = document.createElement('div');
            header.style.marginBottom = '15px';
            header.style.paddingBottom = '15px';
            header.style.borderBottom = '2px solid #667eea';
            header.innerHTML = `
                <strong>Channel:</strong> ${{metadata.channel}}<br>
                <strong>Messages:</strong> ${{metadata.messages}}<br>
                <strong>Authors:</strong> ${{metadata.authors.join(', ')}}<br>
                <strong>Time Range:</strong> ${{metadata.chunk_start}} - ${{metadata.chunk_end}}
            `;
            container.appendChild(header);

            // Load messages on-demand from embeddingsData
            const messages = (embeddingsData[metadata.channel] && embeddingsData[metadata.channel][metadata.chunk_index]) || [];

            if (messages.length === 0) {{
                container.innerHTML += '<p style="color: #999;">No messages available</p>';
                return;
            }}

            messages.forEach((msg, idx) => {{
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message';

                const author = document.createElement('div');
                author.className = 'message-author';
                author.textContent = msg.author;

                const time = document.createElement('div');
                time.className = 'message-time';
                time.textContent = 'GMT+10: ' + formatTimestamp(msg.timestamp);

                const content = document.createElement('div');
                content.className = 'message-content';
                content.textContent = msg.content;

                msgDiv.appendChild(author);
                msgDiv.appendChild(time);
                msgDiv.appendChild(content);
                container.appendChild(msgDiv);
            }});

            document.getElementById('chatbox').classList.add('active');
        }}

        function createPlot(vectorType, method, channel, author) {{
            const allData = tracesData[vectorType][method];

            // Filter by author and channel
            let data = allData;
            if (author !== 'all' || channel !== 'all') {{
                data = allData.filter(point => {{
                    const matchesAuthor = author === 'all' || point.authors.includes(author);
                    const matchesChannel = channel === 'all' || point.channel === channel;
                    return matchesAuthor && matchesChannel;
                }});
            }}

            // Group by cluster
            const clusters = {{}};
            for (const point of data) {{
                const clusterId = point.cluster;
                if (!clusters[clusterId]) {{
                    clusters[clusterId] = [];
                }}
                clusters[clusterId].push(point);
            }}

            // Create traces for each cluster
            const traces = [];
            const uniqueClusters = Object.keys(clusters).map(Number).sort((a, b) => a - b);

            for (let i = 0; i < uniqueClusters.length; i++) {{
                const clusterId = uniqueClusters[i];
                const clusterData = clusters[clusterId];
                const color = clusterColors[i % clusterColors.length];

                const x = clusterData.map(p => p.x);
                const y = clusterData.map(p => p.y);
                const text = clusterData.map(p => {{
                    const meta = p.metadata;
                    const authors = meta.authors.join(', ');
                    return `<b>${{meta.channel}}</b><br>` +
                           `Topic: ${{meta.topic_short}}<br>` +
                           `Technical Topic: ${{meta.technical_topic_short}}<br>` +
                           `Sentiment: ${{meta.sentiment_short}}<br>` +
                           `<br>` +
                           `Authors: ${{authors}}<br>` +
                           `Messages: ${{meta.messages}}<br>` +
                           `Cluster: ${{p.cluster}}`;
                }});

                traces.push({{
                    x: x,
                    y: y,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 10,
                        color: color,
                        opacity: 1
                    }},
                    text: text,
                    hovertemplate: '%{{text}}<extra></extra>',
                    customdata: clusterData.map(p => p.metadata),
                }});
            }}

            const layout = {{
                title: `${{vectorType.replace(/_/g, ' ').toUpperCase()}} - ${{method.toUpperCase()}}${{author === 'all' ? '' : ` - Author: ${{author}}`}}${{channel === 'all' ? '' : ` - Channel: ${{channel}}`}}`,
                hovermode: 'closest',
                plot_bgcolor: '#f8f9fa',
                paper_bgcolor: 'white',
                xaxis: {{
                    title: `${{method.toUpperCase()}} Dimension 1`,
                    showgrid: true,
                    gridwidth: 1,
                    gridcolor: '#e0e0e0'
                }},
                yaxis: {{
                    title: `${{method.toUpperCase()}} Dimension 2`,
                    showgrid: true,
                    gridwidth: 1,
                    gridcolor: '#e0e0e0'
                }},
                height: 600,
                margin: {{ l: 80, r: 20, t: 60, b: 80 }},
                showlegend: false
            }};

            Plotly.newPlot('plotDiv', traces, layout, {{responsive: true}});

            // Attach click handler using Plotly's .on() method
            var myDiv = document.getElementById('plotDiv');
            myDiv.on('plotly_click', function(data) {{
                console.log('plotly_click fired!', data);
                if (data.points && data.points.length > 0) {{
                    const pt = data.points[0];
                    console.log('Got point:', pt);
                    if (pt.customdata) {{
                        displayMessages(pt.customdata);
                    }}
                }}
            }});
        }}

        // Initialize plot
        let currentVector = 'combined';
        let currentMethod = 'umap';
        let currentChannel = 'all';
        let currentAuthor = 'all';

        function updatePlot() {{
            createPlot(currentVector, currentMethod, currentChannel, currentAuthor);
        }}

        document.getElementById('vectorSelect').addEventListener('change', (e) => {{
            currentVector = e.target.value;
            updatePlot();
        }});

        document.getElementById('methodSelect').addEventListener('change', (e) => {{
            currentMethod = e.target.value;
            updatePlot();
        }});

        document.getElementById('channelSelect').addEventListener('change', (e) => {{
            currentChannel = e.target.value;
            updatePlot();
        }});

        document.getElementById('authorSelect').addEventListener('change', (e) => {{
            currentAuthor = e.target.value;
            updatePlot();
        }});

        // Create initial plot
        updatePlot();
    </script>
</body>
</html>"""

    return html


def main():
    print("=" * 80)
    print("Discord Conversation Embedding Visualization")
    print("=" * 80)

    # Load embeddings
    print("\n1. Loading embeddings...")
    embeddings_by_channel = load_embeddings()
    if not embeddings_by_channel:
        print("No embeddings found in output/ directory")
        return

    # Extract vectors and metadata
    print("\n2. Extracting vector data...")
    data = extract_vector_data(embeddings_by_channel)
    vectors = data['vectors']
    metadata = data['metadata']
    print(f"   Total chunks: {len(metadata)}")

    # Reduce and cluster
    print("\n3. Dimensionality reduction and clustering...")
    results = reduce_and_cluster(vectors, metadata)

    # Generate HTML
    print("\n4. Generating HTML visualization...")
    generate_html(results, metadata, embeddings_by_channel)


if __name__ == "__main__":
    main()
