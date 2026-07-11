import os

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise RuntimeError("Переменная MISTRAL_API_KEY не найдена в файле .env")

client = Mistral(api_key=api_key)

response = client.chat.complete(
    model="mistral-medium-latest",
    messages=[
        {
            "role": "user",
            "content": "Ответь одним предложением: что такое RAG?",
        }
    ],
)

print(response.choices[0].message.content)
