import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY", "fw_QGZ3KryaAtGQXoX8myZbLD"),
    base_url="https://api.fireworks.ai/inference/v1",
)

models = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/gemma-4-26b-a4b-it",
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
]

async def main():
    for m in models:
        try:
            print(f"Testing {m}...")
            response = await client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "HI"}],
                max_tokens=100,
                temperature=0,
            )
            print(f"SUCCESS: {m}\n")
        except Exception as e:
            print(f"ERROR: {m} - {e}\n")

asyncio.run(main())
