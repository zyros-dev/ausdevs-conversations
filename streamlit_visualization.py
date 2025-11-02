"""
Interactive Streamlit app for Discord conversation visualization.

Run with:
  streamlit run streamlit_visualization.py
"""

import streamlit as st
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from streamlit_echarts import st_echarts

# Page config
st.set_page_config(page_title="AusDevs Conversation Analysis", layout="wide")

# Database path
DB_PATH = "conversation_data.db"


@st.cache_data
def get_db_options():
    """Get all unique values for filters from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT vector_type FROM chunks ORDER BY vector_type")
    vector_types = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT method FROM chunks ORDER BY method")
    methods = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT channel FROM chunks ORDER BY channel")
    channels = ["All Channels"] + [row[0] for row in cursor.fetchall()]

    cursor.execute('''
        SELECT author, COUNT(*) as count
        FROM messages
        GROUP BY author
        ORDER BY count DESC
        LIMIT 200
    ''')
    authors = ["All Authors"] + [row[0] for row in cursor.fetchall()]

    conn.close()
    return vector_types, methods, channels, authors


@st.cache_data
def load_chunk_data(vector_type: str, method: str, channel: str, min_messages: int):
    """Load and filter chunk data from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = '''
        SELECT id, channel, chunk_index, x, y, authors, msg_count,
               chunk_start, chunk_end, topic_short, technical_topic_short, sentiment_short
        FROM chunks
        WHERE vector_type = ? AND method = ? AND msg_count >= ?
    '''
    params = [vector_type, method, min_messages]

    if channel != "All Channels":
        query += " AND channel = ?"
        params.append(channel)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    # Parse data
    df_data = []
    coords = []

    for row in rows:
        chunk_id, channel_name, chunk_idx, x, y, authors_json, msg_count, start, end, topic_s, tech_s, sent_s = row
        authors_list = json.loads(authors_json)

        df_data.append({
            'chunk_id': chunk_id,
            'channel': channel_name,
            'x': x,
            'y': y,
            'authors': ', '.join(authors_list),
            'messages': msg_count,
            'start': start,
            'end': end,
            'topic': topic_s or "N/A",
            'technical_topic': tech_s or "N/A",
            'sentiment': sent_s or "N/A",
        })
        coords.append([x, y])

    if not coords:
        return pd.DataFrame()

    coords = np.array(coords)
    df = pd.DataFrame(df_data)

    # DBSCAN clustering with recursive splitting for large clusters
    clusterer = DBSCAN(eps=0.5, min_samples=2)
    labels = clusterer.fit_predict(coords)

    def split_large_clusters(labels, coords, max_size=250):
        """Recursively split clusters until all are <= max_size using KMeans."""
        from sklearn.cluster import KMeans

        split_labels = labels.copy()
        next_label = max(labels) + 1 if len(labels) > 0 else 0
        iterations = 0
        max_iterations = 20

        while iterations < max_iterations:
            iterations += 1
            large_clusters = []

            for cluster_id in set(split_labels):
                if cluster_id == -1:
                    continue
                mask = split_labels == cluster_id
                cluster_size = np.sum(mask)
                if cluster_size > max_size:
                    large_clusters.append((cluster_id, np.where(mask)[0], cluster_size))

            if not large_clusters:
                break

            for cluster_id, indices, size in large_clusters:
                sub_coords = coords[indices]
                n_clusters = max(2, (size + max_size - 1) // max_size)

                try:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
                    sub_labels = kmeans.fit_predict(sub_coords)

                    for sub_id in range(n_clusters):
                        sub_mask = sub_labels == sub_id
                        split_labels[indices[sub_mask]] = next_label
                        next_label += 1
                except:
                    split_labels[indices] = next_label
                    next_label += 1

        return split_labels

    split_labels = split_large_clusters(labels, coords, max_size=250)
    df['cluster'] = split_labels

    return df


@st.cache_data
def display_chunk(chunk_id: int) -> str:
    """Fetch and display messages for a chunk."""
    if chunk_id is None:
        return ""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT channel, chunk_start, chunk_end, msg_count FROM chunks WHERE id = ? LIMIT 1', (chunk_id,))
    chunk_info = cursor.fetchone()

    if not chunk_info:
        cursor.close()
        conn.close()
        return "Chunk not found"

    channel, start, end, msg_count = chunk_info

    cursor.execute('SELECT author, content, timestamp FROM messages WHERE chunk_id = ? ORDER BY id', (chunk_id,))
    messages = cursor.fetchall()
    cursor.close()
    conn.close()

    output = f"**Channel:** {channel}\n**Time:** {start} → {end}\n**Messages:** {msg_count}\n\n---\n\n"
    for author, content, timestamp in messages:
        output += f"**{author}** ({timestamp})\n\n{content}\n\n---\n\n"

    return output


def main():
    st.title("AusDevs Conversation Analysis")
    st.markdown("Explore 8,000+ conversation chunks from the AusDevs Discord server. Click a point to see the conversation.")

    if not Path(DB_PATH).exists():
        st.error(f"{DB_PATH} not found!")
        return

    # Get filter options
    vector_types, methods, channels, authors = get_db_options()

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        vector_type = st.selectbox("Vector Type", vector_types)
        method = st.selectbox("Reduction Method", methods)
        channel = st.selectbox("Channel", channels)
        author = st.selectbox("Author", authors)
        min_messages = st.slider("Min Messages", 1, 50, 5)
        search_text = st.text_input("Search descriptions", "")

    # Load data
    df = load_chunk_data(vector_type, method, channel, min_messages)

    if df.empty:
        st.warning("No data matching filters")
        return

    # Apply author and search filters in Python (faster than caching with these params)
    if author != "All Authors" or search_text:
        mask = pd.Series([True] * len(df), index=df.index)

        if author != "All Authors":
            mask = mask & df['authors'].str.contains(author, na=False)

        if search_text:
            search_lower = search_text.lower()
            mask = mask & (
                df['topic'].str.lower().str.contains(search_lower, na=False) |
                df['technical_topic'].str.lower().str.contains(search_lower, na=False) |
                df['sentiment'].str.lower().str.contains(search_lower, na=False)
            )

        df = df[mask]

    if df.empty:
        st.warning("No data matching filters")
        return


    # Prepare data for ECharts (GPU-accelerated with WebGL)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Create scatter data with chunk metadata embedded
    scatter_data = []
    for idx, row in df.iterrows():
        cluster_color = colors[row['cluster'] % len(colors)]
        # Opacity based on cluster size to avoid domination
        opacity = 0.3 if row['cluster'] > 500 else 0.6
        # Store all chunk data in value array for tooltip and click handling
        # Format: [x, y, chunk_id, channel, topic, technical_topic, sentiment, messages, authors]
        scatter_data.append({
            'value': [
                row['x'], row['y'],
                int(row['chunk_id']),
                row['channel'],
                row['topic'],
                row['technical_topic'],
                row['sentiment'],
                int(row['messages']),
                row['authors']
            ],
            'itemStyle': {'color': cluster_color, 'opacity': opacity}
        })

    # ECharts configuration with WebGL rendering
    option = {
        'backgroundColor': '#111111',
        'textStyle': {'color': '#cccccc'},
        'title': {
            'text': f"{vector_type.replace('_', ' ').title()} - {method.upper()} ({len(df)} chunks)",
            'textStyle': {'color': '#cccccc'},
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'item',
            'backgroundColor': 'rgba(0, 0, 0, 0.9)',
            'borderColor': '#666',
            'borderWidth': 1,
            'textStyle': {'color': '#fff', 'fontSize': 12},
            'confine': True,
            'padding': [10, 15],
            'formatter': "function(params){var v=params.value;return 'Chunk '+v[2]+'<br/>Channel: '+v[3]+'<br/>Topic: '+v[4]+'<br/>Technical: '+v[5]+'<br/>Sentiment: '+v[6]+'<br/>Messages: '+v[7]+'<br/>Authors: '+v[8];}"
        },
        'grid': {'left': '10%', 'right': '10%', 'top': '15%', 'bottom': '10%'},
        'xAxis': {
            'type': 'value',
            'name': f"{method.upper()} Dim 1",
            'nameTextStyle': {'color': '#cccccc'},
            'axisLine': {'lineStyle': {'color': '#333'}},
            'axisLabel': {'color': '#999'},
            'splitLine': {'lineStyle': {'color': '#222'}}
        },
        'yAxis': {
            'type': 'value',
            'name': f"{method.upper()} Dim 2",
            'nameTextStyle': {'color': '#cccccc'},
            'axisLine': {'lineStyle': {'color': '#333'}},
            'axisLabel': {'color': '#999'},
            'splitLine': {'lineStyle': {'color': '#222'}}
        },
        'series': [{
            'type': 'scatter',
            'name': 'Chunks',
            'symbolSize': 5,
            'data': scatter_data,
            'itemStyle': {'opacity': 0.6}
        }],
        'animation': False  # Disable animation for large datasets
    }

    # Display chart
    st.markdown("### Visualization")

    # Set up click event handler to return chunk_id from value[2]
    events = {
        "click": "function(params) { return params.value[2]; }"
    }
    chunk_id = st_echarts(option, height=600, events=events)

    # Handle clicks - chunk_id will be the returned value from the click event
    if chunk_id is not None:
        try:
            chunk_id_int = int(chunk_id)
            st.markdown("### Conversation")
            conversation_text = display_chunk(chunk_id_int)
            if conversation_text:
                st.markdown(conversation_text)
            else:
                st.error(f"No data found for chunk {chunk_id_int}")
        except (ValueError, TypeError) as e:
            st.error(f"Error processing click: {e}. Got value: {chunk_id}")


if __name__ == "__main__":
    main()
