# main.py

import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from jose import jwt, JWTError

from services.intent_service import extract_intent, detect_attribute
from services.data_service import resolve_entity, format_attribute_answer
from services.memory_service import save_message, get_recent_messages
from services.rag_service import get_rag_context, get_rag_items
from services.llm_service import answer_with_ai

app = FastAPI()
JWT_ALGORITHM = "HS256"


@app.get("/health")
async def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/ask")
async def ask_ai(req: AskRequest, authorization: str = Header(None)):

    # --------------------------------------------------
    # AUTH 
    # --------------------------------------------------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ", 1)[1].strip()
    JWT_SECRET = os.getenv("JWT_SECRET")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = str(payload["user_id"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # --------------------------------------------------
    # INPUT
    # --------------------------------------------------
    query = req.query.strip()
    q_lower = query.lower()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    await save_message(user_id, "user", query)

    # --------------------------------------------------
    # GREETING / SELF INTRO (STRICT)
    # --------------------------------------------------
    INTRO_TRIGGERS = {
        "hi", "hello", "hey",
        "who are you",
        "what can you do",
        "introduce yourself",
        "tell me about yourself",
        "about yourself",
    }

    if q_lower in INTRO_TRIGGERS:
        answer = (
            "Hi! I’m Anvi AI, a Nashik-based travel assistant. "
            "I help you find hotels, budget stays, luxury resorts, villas, "
            "and amenities in Nashik."
        )
        await save_message(user_id, "assistant", answer)
        return {"answer": answer, "cards": []}

    # --------------------------------------------------
    # INTENT
    # --------------------------------------------------
    intent = extract_intent(query) or {}
    intent_type = intent.get("type")

    # --------------------------------------------------
    # HARD CATEGORY NORMALIZATION (🔥 CORE FIX 🔥)
    # --------------------------------------------------
    if "category_id" not in intent:
        if "resort" in q_lower:
            intent["category_id"] = "2"
        elif "villa" in q_lower:
            intent["category_id"] = "4"
        elif "hotel" in q_lower or "hotels" in q_lower or "stay" in q_lower:
            intent["category_id"] = "1"

    if "budget" in q_lower or "cheap" in q_lower:
        intent["price_segment"] = "budget"

    if "luxury" in q_lower or "premium" in q_lower or "5 star" in q_lower:
        intent["price_segment"] = "luxury"

    # --------------------------------------------------
    # 1️⃣ ENTITY LOOKUP (NAME ONLY / TELL / SHOW / FIND)
    # --------------------------------------------------
    if intent_type == "entity_lookup":
        entity_name = intent.get("entity_name", "")

        # 🔥 FIRST: fuzzy list-based lookup (MOST IMPORTANT FIX)
        items = await get_rag_items(query, intent)
        entity_data = items[0] if len(items) == 1 else None

        # 🔁 FALLBACK: strict resolver
        if not entity_data and entity_name:
            entity_data = await resolve_entity(entity_name, intent, token=token)

        if entity_data:
            detected_attribute = detect_attribute(query)

            # Attribute-only question
            if detected_attribute:
                value = entity_data.get(detected_attribute)
                answer = format_attribute_answer(
                    entity_data, detected_attribute, value
                )
                await save_message(user_id, "assistant", answer)
                return {"answer": answer, "cards": []}

            # Full entity summary
            entity_context = (
                f"Name: {entity_data.get('vendor_name')}\n"
                f"Rating: {entity_data.get('star_rating')}\n"
                f"Address: {entity_data.get('address')}\n"
                f"Description: {entity_data.get('description')}\n"
            )

            history = await get_recent_messages(user_id)

            answer = await answer_with_ai(
                query=query,
                context=entity_context,
                intent=intent,
                memory=history,
            )

            card = {
                "title": entity_data.get("vendor_name"),
                "subtitle": entity_data.get("area_name"),
                "rating": entity_data.get("star_rating"),
                "address": entity_data.get("address"),
                "description": entity_data.get("description"),
                "image": entity_data.get("image_url"),
                "category_id": entity_data.get("category_id"),
                "table_id": entity_data.get("table_id"),
            }

            await save_message(user_id, "assistant", answer)
            return {"answer": answer, "cards": [card]}

    # --------------------------------------------------
    # 2️⃣ FILTERED SEARCH (HOTELS / RESORTS / VILLAS)
    # --------------------------------------------------
    history = await get_recent_messages(user_id)

    rag_context = await get_rag_context(
        keyword=query,
        session_id=req.session_id or "",
        intent=intent,
    )

    if not rag_context:
        return {
            "answer": "No matching places found for your request.",
            "cards": [],
        }

    answer = await answer_with_ai(
        query=query,
        context=rag_context,
        intent=intent,
        memory=history,
    )

    items = await get_rag_items(query, intent)

    cards = [{
        "title": i.get("vendor_name"),
        "subtitle": i.get("area_name"),
        "rating": i.get("star_rating"),
        "address": i.get("address"),
        "description": i.get("description"),
        "image": i.get("image_url"),
        "category_id": i.get("category_id"),
        "table_id": i.get("table_id"),
    } for i in items[:8]]

    await save_message(user_id, "assistant", answer)
    return {"answer": answer, "cards": cards}
