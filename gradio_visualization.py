"""
Interactive Gradio app for Discord conversation visualization.

Run with:
  python gradio_visualization.py          # Local only
  python gradio_visualization.py --share  # Public share link (72 hours)
"""

import gradio as gr
import sqlite3
import json
import numpy as np
import altair as alt
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.cluster import DBSCAN

# Increase Altair's row limit to handle large datasets
alt.data_transformers.enable("default", max_rows=None)

# Global database connection
DB_PATH = "conversation_data.db"


def load_chunk_data(
    vector_type: str,
    method: str,
    channel: str,
    author: str,
    min_messages: int,
    search_text: str = "",
) -> Tuple[alt.Chart, pd.DataFrame]:
    """
    Load and filter chunk data, compute clusters, and return Altair plot + metadata.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Build query
    query = f'''
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
        return alt.Chart(pd.DataFrame()).mark_point(), pd.DataFrame()

    # Parse data
    df_data = []
    coords = []

    for row in rows:
        chunk_id, channel_name, chunk_idx, x, y, authors_json, msg_count, start, end, topic_s, tech_s, sent_s = row
        authors = json.loads(authors_json)

        # Apply author filter
        if author != "All Authors" and author not in authors:
            continue

        # Apply search filter
        search_text_lower = search_text.lower()
        if search_text and not any(search_text_lower in s.lower() for s in [topic_s or "", tech_s or "", sent_s or ""]):
            continue

        df_data.append({
            'chunk_id': chunk_id,
            'channel': channel_name,
            'x': x,
            'y': y,
            'authors': ', '.join(authors),
            'messages': msg_count,
            'start': start,
            'end': end,
            'topic': topic_s or "N/A",
            'technical_topic': tech_s or "N/A",
            'sentiment': sent_s or "N/A",
        })
        coords.append([x, y])

    if not coords:
        return alt.Chart(pd.DataFrame()).mark_point(), pd.DataFrame()

    coords = np.array(coords)
    df = pd.DataFrame(df_data)

    # Cluster with DBSCAN
    clusterer = DBSCAN(eps=0.5, min_samples=2)
    labels = clusterer.fit_predict(coords)

    # Split large clusters (max 250 points per sub-cluster)
    split_labels = labels.copy()
    max_cluster_size = 250
    unique_labels = set(labels)

    next_label = max(labels) + 1 if len(labels) > 0 else 0

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points
            continue
        mask = labels == cluster_id
        cluster_size = np.sum(mask)

        if cluster_size > max_cluster_size:
            # Split this cluster into sub-clusters
            cluster_indices = np.where(mask)[0]
            cluster_coords = coords[cluster_indices]

            # Use DBSCAN again with tighter parameters on this cluster
            sub_clusterer = DBSCAN(eps=0.3, min_samples=2)
            sub_labels = sub_clusterer.fit_predict(cluster_coords)

            # Assign new labels to split clusters
            for sub_label in set(sub_labels):
                if sub_label == -1:
                    continue
                sub_mask = sub_labels == sub_label
                split_indices = cluster_indices[sub_mask]
                split_labels[split_indices] = next_label
                next_label += 1

    df['cluster'] = split_labels

    # Category10 colormap (Altair's built-in equivalent to matplotlib's tab10)
    colors = alt.Scale(scheme='category10')

    # Create selection for visual feedback on click
    selection = alt.selection_point(on='click', nearest=True, fields=['chunk_id'])

    chart = alt.Chart(df).mark_circle(size=100, opacity=0.8).encode(
        x=alt.X('x:Q', scale=alt.Scale(zero=False), axis=alt.Axis(labelColor='#cccccc', titleColor='#cccccc')),
        y=alt.Y('y:Q', scale=alt.Scale(zero=False), axis=alt.Axis(labelColor='#cccccc', titleColor='#cccccc')),
        color=alt.condition(
            selection,
            alt.Color('cluster:N', scale=colors, legend=None),
            alt.value('#333333')
        ),
        tooltip=[
            'chunk_id:N',
            'channel:N',
            'topic:N',
            'technical_topic:N',
            'sentiment:N',
            'authors:N',
            'messages:N'
        ],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.4)),
        strokeWidth=alt.condition(selection, alt.value(2), alt.value(0))
    ).properties(
        width=900,
        height=600,
        title=f"{vector_type.replace('_', ' ').title()} - {method.upper()} ({len(df)} chunks)",
        background='#111111'
    ).add_params(
        selection
    ).interactive()

    return chart, df


def display_chunk(chunk_id: int) -> str:
    """Fetch and display full messages for a chunk."""
    if chunk_id is None:
        return "Click on a point to see messages"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get chunk info
    cursor.execute(
        'SELECT channel, chunk_index, msg_count, chunk_start, chunk_end FROM chunks WHERE id = ? LIMIT 1',
        (chunk_id,)
    )
    chunk_info = cursor.fetchone()

    if not chunk_info:
        return "Chunk not found"

    channel, chunk_idx, msg_count, start, end = chunk_info

    # Get messages
    cursor.execute(
        '''SELECT author, content, timestamp FROM messages WHERE chunk_id = ? ORDER BY id''',
        (chunk_id,)
    )
    messages = cursor.fetchall()
    conn.close()

    # Format output
    output = f"**Channel:** {channel}\n\n"
    output += f"**Time:** {start} → {end}\n\n"
    output += f"**Messages:** {msg_count}\n\n"
    output += "---\n\n"

    for author, content, timestamp in messages:
        output += f"**{author}** ({timestamp})\n\n{content}\n\n---\n\n"

    return output


def init_app():
    """Initialize the Gradio app."""

    # Check if database exists
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found! Run 'python build_database.py' first to create the database."
        )

    # Get filter options from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT vector_type FROM chunks ORDER BY vector_type")
    vector_types = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT method FROM chunks ORDER BY method")
    methods = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT channel FROM chunks ORDER BY channel")
    channels = ["All Channels"] + [row[0] for row in cursor.fetchall()]

    # Get top authors
    cursor.execute('''
        SELECT author, COUNT(*) as count
        FROM messages
        GROUP BY author
        ORDER BY count DESC
        LIMIT 200
    ''')
    authors = ["All Authors"] + [row[0] for row in cursor.fetchall()]

    conn.close()

    # Create interface
    with gr.Blocks(title="AusDevs Conversation Analysis") as demo:
        gr.Markdown(
            """
            # AusDevs Conversation Analysis

            Explore 8,000+ conversation chunks from the AusDevs Discord server.
            Select filters and click on points to see full message content.
            """
        )

        with gr.Row():
            vector_select = gr.Dropdown(
                choices=vector_types,
                value=vector_types[0],
                label="Vector Type",
                interactive=True,
            )
            method_select = gr.Dropdown(
                choices=methods,
                value=methods[0],
                label="Reduction Method",
                interactive=True,
            )

        with gr.Row():
            channel_select = gr.Dropdown(
                choices=channels,
                value="All Channels",
                label="Channel",
                interactive=True,
            )
            author_select = gr.Dropdown(
                choices=authors,
                value="All Authors",
                label="Author",
                interactive=True,
            )

        with gr.Row():
            min_messages = gr.Slider(
                minimum=1,
                maximum=50,
                value=5,
                step=1,
                label="Min Messages per Chunk",
                interactive=True,
            )
            search_text = gr.Textbox(
                placeholder="Search in descriptions...",
                label="Search",
                interactive=True,
            )

        # Plot and messages
        plot = gr.Plot(label="Visualization")
        messages_display = gr.Markdown(value="Click on a point to see messages")

        # State to store the dataframe for click handling
        df_state = gr.State(pd.DataFrame())

        # Update on any filter change
        def update_plot(*args):
            vector_type, method, channel, author, min_msg, search = args
            chart, df = load_chunk_data(
                vector_type=vector_type,
                method=method,
                channel=channel,
                author=author,
                min_messages=min_msg,
                search_text=search,
            )
            return chart, df

        vector_select.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )
        method_select.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )
        channel_select.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )
        author_select.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )
        min_messages.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )
        search_text.change(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state]
        )

        # Note: Gradio's gr.Plot doesn't expose click events directly.
        # Users can see chunk_id in the tooltip by hovering over points.
        # For now, clicking functionality would require custom JavaScript.

        # Load initial plot
        demo.load(
            update_plot,
            inputs=[vector_select, method_select, channel_select, author_select, min_messages, search_text],
            outputs=[plot, df_state],
        )

    return demo


if __name__ == "__main__":
    demo = init_app()
    demo.launch()
