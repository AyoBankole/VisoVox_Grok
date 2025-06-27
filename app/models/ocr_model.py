from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text(image_url: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Extract the text from this image."}
            ]}
        ]
    )
    return response.choices[0].message.content.strip()