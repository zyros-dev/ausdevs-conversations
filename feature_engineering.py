"""
Feature Engineering Pipeline for Discord Conversations

Extracts conversation chunks from Discord exports and generates:
- Topic vectors (what the conversation was about)
- Energy vectors (tone/sentiment)
- Summary vectors (semantic representation)

Architecture:
1. Chunker: Groups messages into coherent conversations
2. DescriberLLM: Analyzes chunks to extract features
3. Embedder: Converts feature descriptions to vectors
"""

import json
import os
import time
import logging
import asyncio
import random
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    AsyncRetrying
)

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Async retry for API calls with exponential backoff and jitter
def async_retry_with_backoff(func):
    """Decorator for async API calls with exponential backoff (with jitter) and 50 retries."""
    async def wrapper(*args, **kwargs):
        last_exception = None
        for attempt_num in range(50):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt_num < 49:  # Don't sleep on last attempt
                    base_wait = min(60, (2 ** attempt_num) * 1)  # Exponential backoff: 1, 2, 4, 8... up to 60s
                    jitter = random.uniform(0.5, 1.5)  # ±50% variance to avoid thundering herd
                    wait_time = base_wait * jitter
                    await asyncio.sleep(wait_time)
        raise last_exception
    return wrapper


@dataclass
class Message:
    """Represents a single Discord message."""
    author: str
    content: str
    timestamp: str


@dataclass
class Channel:
    """Represents a Discord channel."""
    name: str
    id: str


@dataclass
class Chunk:
    """Represents a coherent conversation chunk."""
    messages: List[Message]
    channel: Channel
    start_time: str
    end_time: str

    @property
    def text(self) -> str:
        """Return formatted text of all messages in chunk."""
        return "\n".join([f"{m.author}: {m.content}" for m in self.messages])

    @property
    def message_count(self) -> int:
        return len(self.messages)


class ChunkAnalysis(BaseModel):
    """
    Structured analysis of a conversation chunk.

    Six strings: three detailed (50-200 words) and three short (1-5 words).
    """
    topic: str = Field(
        description="Comprehensive summary of what was discussed and important outcomes (50-200 words)"
    )
    topic_short: str = Field(
        description="Very short summary of discussion/outcomes (1-5 words only)"
    )
    technical_topic: str = Field(
        description="The specific company, product, algorithm, or CS concept being discussed (50-200 words)"
    )
    technical_topic_short: str = Field(
        description="Very short name of the company/algorithm/CS topic (1-5 words only)"
    )
    sentiment: str = Field(
        description="Description of the tone, mood, sentiment and emotional undercurrents (50-200 words)"
    )
    sentiment_short: str = Field(
        description="Very short summary of sentiment/energy (1-5 words only)"
    )


@dataclass
class ChunkDescription:
    """Pairs a chunk with its detailed analysis."""
    chunk: Chunk
    analysis: ChunkAnalysis


class DescriberLLM:
    """Uses Gemini to analyze chunks and extract features with structured output."""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-lite"):
        """Initialize with Gemini API.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (default: gemini-2.5-flash)
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env or parameters")

        genai.configure(api_key=api_key)
        # Use structured output with JSON schema
        self.model = genai.GenerativeModel(
            model,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ChunkAnalysis,
                max_output_tokens=8192,  # 6 fields × ~200 words each ≈ 1600 tokens + padding
            ),
        )
        self.model_name = model

    @async_retry_with_backoff
    async def should_merge(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        prompt = f"""Analyze these two conversation chunks and determine if they should be merged because they're about the same topic/discussion.

Consider:
- Are they discussing the same company, product, or concept?
- Is the conversation continuing naturally or did it shift to something new?
- Do the speakers seem to be addressing the same discussion thread?

CHUNK 1 (last 2 messages):
{chr(10).join([f"{msg.author}: {msg.content}" for msg in chunk1.messages[-2:]])}

CHUNK 2 (first 2 messages):
{chr(10).join([f"{msg.author}: {msg.content}" for msg in chunk2.messages[:2]])}

Reply with ONLY one word: YES or NO"""

        # Use simple model for lightweight decision
        simple_model = genai.GenerativeModel(self.model_name)
        response = await simple_model.generate_content_async(prompt)
        return "YES" in response.text.upper()

    @async_retry_with_backoff
    async def describe(self, chunk: Chunk) -> ChunkDescription:
        """
        Extract detailed and short descriptions from a chunk using structured output.

        Returns ChunkDescription with ChunkAnalysis containing:
        - topic: Comprehensive summary of discussion and outcomes (50-200 words)
        - topic_short: Very short summary of discussion/outcomes (1-5 words)
        - technical_topic: The specific company/algorithm/CS concept (50-200 words)
        - technical_topic_short: Company/algorithm/CS topic name (1-5 words)
        - sentiment: The tone/mood/sentiment of the conversation (50-200 words)
        - sentiment_short: Very short summary of sentiment/energy (1-5 words)
        """
        prompt = f"""Analyze this Discord conversation chunk and extract both detailed and short descriptions for:
1. Topic: comprehensive summary of what was discussed and important outcomes
2. Technical Topic: the specific company, product, algorithm, or CS concept being discussed
3. Sentiment: the tone, mood, and emotional undercurrents

For each, provide:
- A detailed 50-200 word description
- A very short 1-5 word summary

Conversation:
{chunk.text}"""

        response = await self.model.generate_content_async(prompt)

        try:
            analysis_dict = json.loads(response.text)
            analysis = ChunkAnalysis(**analysis_dict)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Response text: {response.text}")
            raise

        return ChunkDescription(
            chunk=chunk,
            analysis=analysis
        )


class Chunker:
    def __init__(self, describer_llm: DescriberLLM, channel: Channel = None):
        self.describer = describer_llm
        self.channel = channel
        self.max_chunk_messages = 50
        self.max_chunk_time_minutes = 60

    async def chunk(self, messages: List[Message]) -> List[Chunk]:
        if not messages:
            return []

        rough_chunks = self._heuristic_chunk(messages)
        print(f"[Chunker] Created {len(rough_chunks)} rough chunks from {len(messages)} messages")

        final_chunks = await self._merge_adjacent_chunks(rough_chunks)
        print(f"[Chunker] Merged to {len(final_chunks)} final chunks")

        return final_chunks

    def _heuristic_chunk(self, messages: List[Message]) -> List[Chunk]:
        chunks = []
        current_chunk = []

        for msg in messages:
            current_chunk.append(msg)

            # Check if we should start a new chunk
            should_split = False

            if len(current_chunk) >= self.max_chunk_messages:
                should_split = True
            elif len(current_chunk) > 1:
                time_diff = self._time_diff_minutes(current_chunk[0].timestamp, msg.timestamp)
                if time_diff > self.max_chunk_time_minutes:
                    should_split = True

            if should_split:
                chunks.append(self._create_chunk(current_chunk[:-1]))
                current_chunk = [msg]

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk))

        return chunks

    def _create_chunk(self, messages: List[Message]) -> Chunk:
        return Chunk(
            messages=messages,
            channel=self.channel,
            start_time=messages[0].timestamp,
            end_time=messages[-1].timestamp
        )

    async def _merge_adjacent_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks

        # Batch check all adjacent pairs concurrently
        merge_decisions = []
        tasks = []
        for i in range(1, len(chunks)):
            tasks.append(self.describer.should_merge(chunks[i-1], chunks[i]))

        # Run all merge checks in parallel
        merge_decisions = await asyncio.gather(*tasks, return_exceptions=True)

        # Build merged chunks based on decisions
        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            decision = merge_decisions[i-1]
            if isinstance(decision, Exception):
                # On error, don't merge
                merged.append(chunks[i])
            elif decision:
                # Merge with previous chunk
                merged[-1] = Chunk(
                    messages=merged[-1].messages + chunks[i].messages,
                    channel=merged[-1].channel,
                    start_time=merged[-1].start_time,
                    end_time=chunks[i].end_time
                )
            else:
                merged.append(chunks[i])

        return merged

    @staticmethod
    def _time_diff_minutes(ts1: str, ts2: str) -> float:
        """Calculate minutes between two ISO 8601 timestamps."""
        try:
            dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
            return (dt2 - dt1).total_seconds() / 60
        except Exception:
            return 0


class Embedder:
    def __init__(self, api_key: str = None, model: str = "models/embedding-001"):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env or parameters")

        genai.configure(api_key=api_key)
        self.model = model

    async def embed(self, descriptions: List[ChunkDescription]) -> List[dict]:
        # Batch embed all texts at once for efficiency
        all_texts = []
        text_to_desc = {}  # Map text -> description index

        for i, desc in enumerate(descriptions):
            analysis = desc.analysis
            all_texts.extend([analysis.topic, analysis.technical_topic, analysis.sentiment])
            text_to_desc[len(all_texts) - 3:len(all_texts)] = i

        # Embed in batches of 256 with 32 concurrent requests
        embeddings_dict = {}  # Map text -> embedding
        batch_size = 256
        semaphore = asyncio.Semaphore(32)

        async def embed_batch(batch_start, batch_end):
            async with semaphore:
                batch_texts = all_texts[batch_start:batch_end]
                batch_embeddings = await self._embed_texts_batch(batch_texts)
                for text, emb in zip(batch_texts, batch_embeddings):
                    embeddings_dict[text] = emb

        # Create all batch tasks
        batch_tasks = []
        for batch_start in range(0, len(all_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(all_texts))
            batch_tasks.append(embed_batch(batch_start, batch_end))

        # Run all batches concurrently
        await asyncio.gather(*batch_tasks)

        # Assemble results
        results = []
        for i, desc in enumerate(descriptions):
            analysis = desc.analysis

            # Retrieve embeddings
            topic_vec = embeddings_dict[analysis.topic]
            technical_topic_vec = embeddings_dict[analysis.technical_topic]
            sentiment_vec = embeddings_dict[analysis.sentiment]

            # Combine into single vector (technical_topic drives clustering)
            combined_vec = np.concatenate([topic_vec, technical_topic_vec, sentiment_vec])

            results.append({
                'channel': {
                    'name': desc.chunk.channel.name,
                    'id': desc.chunk.channel.id,
                },
                'messages': [
                    {
                        'author': msg.author,
                        'content': msg.content,
                        'timestamp': msg.timestamp,
                    }
                    for msg in desc.chunk.messages
                ],
                'chunk_start': desc.chunk.start_time,
                'chunk_end': desc.chunk.end_time,
                'topic': analysis.topic,
                'topic_short': analysis.topic_short,
                'technical_topic': analysis.technical_topic,
                'technical_topic_short': analysis.technical_topic_short,
                'sentiment': analysis.sentiment,
                'sentiment_short': analysis.sentiment_short,
                'topic_embedding': topic_vec.tolist(),
                'technical_topic_embedding': technical_topic_vec.tolist(),
                'sentiment_embedding': sentiment_vec.tolist(),
                'combined_embedding': combined_vec.tolist(),
            })

        return results

    @async_retry_with_backoff
    async def _embed_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
        result = await genai.embed_content_async(
            model=self.model,
            content=texts
        )
        return [np.array(emb) for emb in result['embedding']]


# Example usage
if __name__ == "__main__":
    describer = DescriberLLM()
    chunker = Chunker(describer)
    embedder = Embedder()

    print("Feature engineering pipeline ready!")
