from openai import AsyncOpenAI
import asyncio
import os

client = AsyncOpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY", "fw_QGZ3KryaAtGQXoX8myZbLD"),
    base_url="https://api.fireworks.ai/inference/v1",
)

async def call_model(model_id: str, prompt: str):
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0,
            # Let's test with reasoning_effort="none"
            extra_body={"reasoning_effort": "none"} 
            # Note: openai python client doesn't natively have reasoning_effort parameter in older versions, 
            # or maybe it does, but we can pass it in extra_body or directly if supported.
            # I will just pass it directly as the teammate did, to replicate perfectly.
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR: {e}")

async def main():
    try:
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k2p7-code",
            messages=[{"role": "user", "content": "HI"}],
            max_tokens=1000,
            temperature=0,
            reasoning_effort="none",
        )
        print(response)
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(main())
