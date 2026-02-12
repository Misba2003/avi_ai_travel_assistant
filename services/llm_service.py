# services/llm_service.py

import os
from typing import Dict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)


async def answer_with_ai(
    query: str,
    context: str,
    intent: Dict,
    memory: str
) -> str:
    if not context or context.strip() == "":
        return "No matching data found in our listings."

    system_msg = """
You are Anvi AI, a Nashik-based travel assistant.

RULES:
- Use ONLY the data provided in CONTEXT. Do NOT add or invent any entities, places, or categories.
- Phrase and summarize only what is in CONTEXT. Do NOT invent hotels, theaters, amenities, ratings, or locations.
- Respond naturally like a helpful travel assistant.
- Do NOT say phrases like "here is a summary" or "based on the context".
"""

    user_msg = f"""
USER QUERY:
{query}

CONTEXT:
{context}
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        top_p=0.9
    )

    return completion.choices[0].message.content.strip()
