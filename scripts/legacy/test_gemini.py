import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
print(f"Key loaded: {api_key[:8]}... | Model: {model_name}")

client = genai.Client(api_key=api_key)

print(f"\n--- Testing Streaming ({model_name}) ---")
try:
    stream = client.models.generate_content_stream(
        model=model_name,
        contents="Say 'Stream works perfectly!'"
    )
    for chunk in stream:
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n\nSUCCESS!")
except Exception as e:
    print("\nStreaming failed:", e)