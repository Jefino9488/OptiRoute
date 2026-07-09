import asyncio
import os
import json
from app.router.pipeline import RoutingPipeline

# Ensure API key is set for test
if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = "fw_QGZ3KryaAtGQXoX8myZbLD"

async def main():
    print("Initializing RoutingPipeline...")
    pipeline = RoutingPipeline()
    
    print("\nSending prompt: 'hello how are you' (Required accuracy: 0.8)")
    result = await pipeline.route(
        prompt="hello how are you",
        required_accuracy=0.8
    )
    
    print("\n=== ROUTER RESPONSE ===")
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    asyncio.run(main())
