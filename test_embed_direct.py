import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

genai.configure(api_key=api_key)

async def test_embed():
    print("Testing embed_content_async...")
    start = time.time()
    try:
        result = await genai.embed_content_async(
            model="models/embedding-001",
            content=["hello world", "test text"]
        )
        elapsed = time.time() - start
        print(f"Success! Got {len(result['embeddings'])} embeddings in {elapsed:.2f}s")
        print(f"Embedding shape: {len(result['embeddings'][0])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("Starting test...")
asyncio.run(test_embed())
print("Done.")
