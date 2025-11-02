"""
Main script to process a Discord channel through the feature engineering pipeline.

Usage:
    python process_channel.py <path_to_channel.json>
    python process_channel.py ausdevs_again/"game-dev.json"
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
from feature_engineering import Message, Channel, Chunker, DescriberLLM, Embedder, ChunkDescription


def load_channel(filepath: str) -> Tuple[List[Message], Channel]:
    """Load messages from a Discord JSON export."""
    print(f"Loading channel: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        channel_name = data['channel']['name']
        channel_id = data['channel']['id']
        message_count = data['messageCount']

        print(f"  Channel: {channel_name}")
        print(f"  Total messages: {message_count}")

        messages = []
        for msg in data['messages']:
            if msg.get('content'):  # Skip empty messages
                messages.append(Message(
                    author=msg['author']['name'],
                    content=msg['content'],
                    timestamp=msg['timestamp']
                ))

        print(f"  Non-empty messages: {len(messages)}")
        channel = Channel(name=channel_name, id=channel_id)
        return messages, channel


def save_results(embeddings: List[dict], input_filename: str):
    """Save results to output directory as single JSON file."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Use input filename (without .json) + _embeddings.json for easy matching
    base_name = Path(input_filename).stem
    output_file = output_dir / f"{base_name}_embeddings.json"
    with open(output_file, 'w') as f:
        json.dump(embeddings, f, indent=2)
    print(f"  Results saved to {output_file}")


def print_sample_results(embeddings: List[dict], num_samples: int = 3):
    """Print sample results from the pipeline."""
    print(f"\n{'='*80}")
    print(f"SAMPLE RESULTS (first {num_samples} chunks)")
    print(f"{'='*80}\n")

    for i, emb in enumerate(embeddings[:num_samples]):
        print(f"CHUNK {i+1}:")
        print(f"  Messages: {len(emb['messages'])}")
        print(f"  Time range: {emb['chunk_start']} to {emb['chunk_end']}")
        print(f"\n  ANALYSIS:")
        print(f"    Topic: {emb['topic']}")
        print(f"    Technical Topic: {emb['technical_topic']}")
        print(f"    Sentiment: {emb['sentiment']}")
        print(f"\n  EMBEDDINGS:")
        print(f"    Topic vector length: {len(emb['topic_embedding'])}")
        print(f"    Technical Topic vector length: {len(emb['technical_topic_embedding'])}")
        print(f"    Sentiment vector length: {len(emb['sentiment_embedding'])}")
        print(f"    Combined vector length: {len(emb['combined_embedding'])}")
        print()


async def main():
    if len(sys.argv) > 1:
        channel_file = sys.argv[1]
    else:
        # Default to game-dev for testing
        channel_file = 'ausdevs_again/AusDevs 2.0.0 - TECHNOLOGIES - game-dev [1306074417146630225].json'

    # Load messages
    messages, channel = load_channel(channel_file)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    describer = DescriberLLM()
    chunker = Chunker(describer, channel=channel)
    embedder = Embedder()

    # Process pipeline
    print(f"\nStep 1: Chunking {len(messages)} messages...")
    chunks = await chunker.chunk(messages)

    print(f"\nStep 2: Describing {len(chunks)} chunks (asyncio, 4000 concurrent)...")

    # Semaphore to limit concurrent requests to 4000
    semaphore = asyncio.Semaphore(4000)

    async def describe_with_semaphore(chunk):
        async with semaphore:
            return await describer.describe(chunk)

    # Create all tasks concurrently
    describe_tasks = [describe_with_semaphore(chunk) for chunk in chunks]
    descriptions = []

    completed = 0
    for coro in asyncio.as_completed(describe_tasks):
        try:
            desc = await coro
            descriptions.append(desc)
            completed += 1
            if completed % max(1, len(chunks) // 10) == 0:
                print(f"  [{completed}/{len(chunks)}] Progress...")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    print(f"\nStep 3: Embedding {len(descriptions)} descriptions (batch + async)...")
    embeddings = await embedder.embed(descriptions)

    # Save and display results
    print(f"\nSaving results...")
    save_results(embeddings, channel_file)

    print_sample_results(embeddings)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"  Input: {len(messages)} messages")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Vectors: {len(embeddings)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
