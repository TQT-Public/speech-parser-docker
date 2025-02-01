import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# gpt-4o-mini
completion = client.chat.completions.create(
    model="gpt-3.5-turbo", store=True, messages=[{"role": "user", "content": "write a haiku about ai"}]
)

print(completion.choices[0].message)
