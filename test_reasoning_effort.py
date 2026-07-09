from openai import AsyncOpenAI
import asyncio
import os

client = AsyncOpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY", "fw_QGZ3KryaAtGQXoX8myZbLD"),
    base_url="https://api.fireworks.ai/inference/v1",
)

async def main():
    try:
        print("Testing WITHOUT reasoning_effort...")
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k2p7-code",
            messages=[{"role": "user", "content": "HI"}],
            max_tokens=1000,
            temperature=0,
        )
        print("SUCCESS!")
    except Exception as e:
        print(f"ERROR: {e}")
        
    try:
        print("\nTesting gemma-4-26b-a4b-it WITH reasoning_effort...")
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/gemma-4-26b-a4b-it",
            messages=[{"role": "user", "content": "HI"}],
            max_tokens=1000,
            temperature=0,
            reasoning_effort="none",
        )
        print("SUCCESS for Gemma!")
    except Exception as e:
        print(f"ERROR for Gemma: {e}")

asyncio.run(main())
