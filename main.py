# main.py

import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# Load environment variables
# ------------------------
project_root = Path(__file__).resolve().parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

# ------------------------
# Import services
# ------------------------
from services.intent_service import extract_intent, detect_attribute
from services.rag_service import get_rag_context, get_rag_items
from services.llm_service import answer_with_ai
from services.memory_service import add_to_memory, get_memory, get_memory_size
from services.data_service import resolve_entity, format_attribute_answer

# ------------------------
# FastAPI App Setup
# ------------------------
app = FastAPI(title="ANVI AI Backend")
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "ok"}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow Flutter app / mobile / web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Request Model
# ------------------------
class AskRequest(BaseModel):
    query: str
    session_id: str


# ------------------------
# MAIN ENDPOINT
# ------------------------
@app.post("/ask")
async def ask_ai(req: AskRequest):
    try:
        query = req.query.strip()
        session_id = req.session_id.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        print(f"[DEBUG] /ask → {query} | session: {session_id}")

        # 1️⃣ Store USER message in SESSION memory
        mem_size = await add_to_memory(session_id, "user", query)
        print(f"[DEBUG] Memory size after user msg: {mem_size} | session={session_id}")

        # 2️⃣ Extract intent
        intent = extract_intent(query)
        category_keyword = intent["category"]
        print(f"[DEBUG] Intent category={category_keyword} | session={session_id}")

        # 2.5️⃣ Entity + Attribute Bypass (before RAG/LLM)
        if intent.get("type") == "entity_lookup":
            detected_attribute = detect_attribute(query)

            # ✅ Existing: entity + attribute bypass (NO LLM)
            if detected_attribute:
                entity_name = intent.get("entity_name", "")
                if entity_name:
                    print(f"[DEBUG] Entity lookup: {entity_name} | attribute: {detected_attribute} | session={session_id}")
                    entity_data = await resolve_entity(entity_name, intent)

                    if entity_data:
                        value = entity_data.get(detected_attribute)
                        answer = format_attribute_answer(entity_data, detected_attribute, value)

                        # Store assistant reply in memory
                        mem_size = await add_to_memory(session_id, "assistant", answer)
                        print(f"[DEBUG] Entity attribute response. Memory size={mem_size} | session={session_id}")

                        return {
                            "answer": answer,
                            "cards": []
                        }
                    else:
                        # Entity not found, fall through to normal flow
                        print(f"[DEBUG] Entity '{entity_name}' not found, using normal flow | session={session_id}")

            # ✅ NEW: entity-only queries (LLM summary on single resolved entity)
            else:
                entity_name = intent.get("entity_name", "")
                if entity_name:
                    print(f"[DEBUG] Entity-only lookup: {entity_name} | session={session_id}")
                    entity_data = await resolve_entity(entity_name, intent)

                    if entity_data:
                        # Fetch SESSION memory (needed for LLM call)
                        memory = await get_memory(session_id)
                        mem_size = await get_memory_size(session_id)
                        print(f"[DEBUG] Memory size={mem_size} | session={session_id}")

                        amenities = entity_data.get("amenities") or []
                        if isinstance(amenities, list) and amenities:
                            amenities_str = ", ".join(amenities[:12])
                        else:
                            amenities_str = "Not provided"

                        # Minimal single-entity context (no multi-entity fetch)
                        entity_context = (
                            "[1]\n"
                            f"Name: {entity_data.get('name') or 'Unknown'}\n"
                            f"Rating: {entity_data.get('rating') or 'Not provided'}\n"
                            f"Address: {entity_data.get('address') or 'Not provided'}\n"
                            f"Amenities: {amenities_str}\n"
                            "Description: Not provided\n"
                            "----"
                        )

                        summary_query = (
                            f"Tell me something about {entity_data.get('name') or entity_name}. "
                            "Write a short, factual 2–3 sentence summary using ONLY the provided context."
                        )

                        answer = await answer_with_ai(
                            query=summary_query,
                            context=entity_context,
                            intent=intent,
                            memory=memory
                        )

                        # Single card (do not fetch multiple hotels)
                        card = {
                            "title": entity_data.get("name"),
                            "subtitle": "",
                            "rating": entity_data.get("rating"),
                            "address": entity_data.get("address"),
                            "description": "",
                            "image": None
                        }

                        # Store assistant reply in memory
                        mem_size = await add_to_memory(session_id, "assistant", answer)
                        print(f"[DEBUG] Entity-only response. Memory size={mem_size} | session={session_id}")

                        return {
                            "answer": answer,
                            "cards": [card]
                        }
                    # If entity not found → fall through to existing generic behavior

        # 3️⃣ Build RAG context
        context = await get_rag_context(category_keyword, session_id, intent)
        if context:
            print(
                f"[DEBUG] RAG context built (len={len(context)}) | session={session_id}"
            )
        else:
            print(f"[DEBUG] RAG context EMPTY | session={session_id}")

        # 4️⃣ Fetch SESSION memory
        memory = await get_memory(session_id)
        mem_size = await get_memory_size(session_id)
        print(f"[DEBUG] Memory size={mem_size} | session={session_id}")

        # 5️⃣ LLM Answer
        answer = await answer_with_ai(
            query=query,
            context=context or "",
            intent=intent,
            memory=memory
        )

        # 7️⃣ Build UI cards
        items = await get_rag_items(category_keyword, intent)

        cards = []
        for item in items[:8]:
            cards.append({
                "title": item.get("vendor_name"),
                "subtitle": item.get("area_name"),
                "rating": item.get("star_rating"),
                "address": item.get("address"),
                "description": item.get("description"),
                "image": item.get("image_url")
            })

        # 6️⃣ Store ASSISTANT reply in SESSION memory
        mem_size = await add_to_memory(session_id, "assistant", answer)
        print(
            f"[DEBUG] Stored assistant reply. Memory size={mem_size} | session={session_id}"
        )

        return {
            "answer": answer,
            "cards": cards
        }

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR]", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
