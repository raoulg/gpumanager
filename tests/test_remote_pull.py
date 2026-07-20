import httpx
import asyncio
import json
import sys

async def test_remote_pull(model_name="tinyllama"):
    url = "http://145.38.184.153:8000/api/pull"
    print(f"Testing pull for {model_name} at {url}...")
    
    payload = {
        "name": model_name,
        "stream": True
    }
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Content-Type": "application/json"}) as response:
                print(f"Status Code: {response.status_code}")
                
                if response.status_code != 200:
                    print("Error Response:")
                    print(await response.read())
                    return

                print("Streaming progress:")
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            
                            if total > 0:
                                percent = (completed / total) * 100
                                sys.stdout.write(f"\r{status}: {percent:.1f}%")
                            else:
                                sys.stdout.write(f"\r{status}")
                            sys.stdout.flush()
                            
                        except json.JSONDecodeError:
                            print(f"\nInvalid JSON: {line}")
                
                print("\nPull completed!")
                
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "tinyllama"
    asyncio.run(test_remote_pull(model))
