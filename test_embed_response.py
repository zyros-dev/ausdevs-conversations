import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit(1)

genai.configure(api_key=api_key)

async def test_embed():
    print("Testing embed_content_async...")
    try:
        result = await genai.embed_content_async(
            model="models/embedding-001",
            content=["hello world", "test text"]
        )
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        if hasattr(result, '__dict__'):
            print(f"Result dict: {result.__dict__}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_embed())
