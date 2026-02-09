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
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ", 1)[1]
    JWT_SECRET = os.getenv("JWT_SECRET")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = str(payload["user_id"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = req.query.strip()
    q_lower = query.lower()

    await save_message(user_id, "user", query)

    # ---------------- GREETINGS ----------------
    INTRO_TRIGGERS = {
        "hi", "hello", "hey",
        "how are you",
        "how can you help",
        "tell me about yourself",
        "who are you",
        "what can you do",
        "introduce yourself",
        "about yourself"
    }
    if any(t in q_lower for t in INTRO_TRIGGERS):
        answer = (
            "Hi! I’m Anvi AI, a Nashik-based travel assistant. "
            "I help you find hotels, budget stays, luxury resorts, villas, "
            "and amenities in and around Nashik. "
            "You can ask me for hotel details, filters like pool or budget, "
            "or specific questions about any hotel."
        )
        await save_message(user_id, "assistant", answer)
        return {"answer": answer, "cards": []}

    intent = extract_intent(query) or {}
    intent_type = intent.get("type")

    # =========================================================
    # 1️⃣ SPECIFIC HOTEL (entity + attribute)
    # =========================================================
    if intent_type == "entity_lookup":
        entity_name = intent.get("entity_name")
        if entity_name:
            entity_data = await resolve_entity(entity_name, intent, token=token)

            if entity_data:
                detected_attribute = detect_attribute(query)
                if detected_attribute:
                    value = entity_data.get(detected_attribute)
                    answer = format_attribute_answer(
                        entity_data, detected_attribute, value
                    )
                    await save_message(user_id, "assistant", answer)
                    return {"answer": answer, "cards": []}

                entity_context = (
                    f"Hotel Name: {entity_data.get('vendor_name')}\n"
                    f"Rating: {entity_data.get('star_rating')}\n"
                    f"Address: {entity_data.get('address')}\n"
                    f"Amenities: {', '.join(entity_data.get('amenities', []))}\n"
                    f"Description: {entity_data.get('description')}\n"
                )

                history = await get_recent_messages(user_id)

                answer = await answer_with_ai(
                    query=query,
                    context=entity_context,
                    intent=intent,
                    memory=history
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

    # =========================================================
    # 2️⃣ LIST / FILTER SEARCH
    # =========================================================
    if any(w in q_lower for w in ["hotel", "hotels", "stay", "resort", "villa"]):
        history = await get_recent_messages(user_id)

        rag_context = await get_rag_context(
            keyword=query,
            session_id=req.session_id or "",
            intent=intent,
        )

        items = await get_rag_items(query, intent)

        if items:
            answer = await answer_with_ai(
                query=query,
                context=rag_context,
                intent=intent,
                memory=history,
            )

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

    return {"answer": "No matching data found.", "cards": []}
