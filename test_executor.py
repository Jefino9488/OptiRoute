import asyncio
import os
import sys

# We must adjust allowed_models for the test so it doesn't ValueError
from app.config import get_settings
settings = get_settings()
settings.allowed_models["kimi-k2p6"] = "accounts/fireworks/models/kimi-k2p6"

from app.executors.fireworks import FireworksExecutor

async def main():
    executor = FireworksExecutor()
    print("Executing request to kimi-k2p6...")
    try:
        res = await executor.execute(
            prompt="Reply with exactly 'OK'",
            model_id="kimi-k2p6"
        )
        print(f"SUCCESS: {res.response}")
        print(f"Tokens in: {res.tokens_input}, out: {res.tokens_output}")
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    asyncio.run(main())
