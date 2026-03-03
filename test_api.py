import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

print(f"Testing API key: {api_key}")
print(f"Testing model: {model}")

# Test with a simple request
try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, are you working?"}
            ]
        }
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ API is working!")
        print("Response:", response.json())
    else:
        print("❌ API error:")
        print(response.text)
        
except Exception as e:
    print(f"❌ Request failed: {e}")