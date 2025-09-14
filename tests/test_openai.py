import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "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")
response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test"}])
print(response.choices[0].message.content)