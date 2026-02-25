import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def check_models():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    print(f"Testing API: {base_url}")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    try:
        models = await client.models.list()
        print("\nAvailable Models:")
        for m in models.data:
            print(f"- {m.id}")
            
        # Test a simple completion with a common model
        test_model = "gpt-3.5-turbo" if "gpt-3.5-turbo" in [m.id for m in models.data] else models.data[0].id
        print(f"\nTesting completion with: {test_model}...")
        resp = await client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Hello, respond with ONE word."}],
            max_tokens=5
        )
        print(f"Response: {resp.choices[0].message.content}")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(check_models())
