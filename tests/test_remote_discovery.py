import httpx
import asyncio
import json

async def test_remote_discovery():
    url = "http://145.38.184.153:8000/api/tags"
    print(f"Testing {url}...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            else:
                print("Response Text:")
                print(response.text)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_remote_discovery())
