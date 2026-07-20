import httpx
import asyncio
import sys

async def test_list_models(manager_host="145.38.184.153"):
    url = f"http://{manager_host}:8000/api/tags"
    print(f"Testing list_models at {url}...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"Found {len(models)} models.")
            else:
                print("Failed to list models.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_list_models())
