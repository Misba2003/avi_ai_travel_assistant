# main.py

import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Header
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
from services.memory_service import get_recent_messages, save_message
from services.data_service import resolve_entity, format_attribute_answer, normalize_name

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
    session_id: str |None= None
# ------------------------
# MAIN ENDPOINT
# ------------------------
@app.post("/ask")
async def ask_ai(
    req: AskRequest,
    authorization: str = Header(None),
):
    try:
        # 0️⃣ Auth: Require valid Bearer token from caller (Flutter / frontend)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")

        token = authorization.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        app_user_id = token

        query = req.query.strip()
        session_id = (req.session_id or "").strip()


        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        print(f"[DEBUG] /ask → {query} | session: {session_id}")

        # 1️⃣ Store USER message in persistent memory (by app_user_id)
        await save_message(app_user_id, "user", query)
        mem_size = 0
        print("[DEBUG] Stored user message in PostgreSQL memory")

        # ------------------------
        # Conversational short-circuit (NO RAG, NO LLM, NO DATA)
        # ------------------------
        q_lower = query.lower().strip()

        CONVERSATIONAL_KEYWORDS = {
            "hi", "hello", "hey",
            "good morning", "good evening", "good afternoon",
            "what can you help me with", "what can you do",
            "how can you help me", "what do you do"
        }

        DOMAIN_KEYWORDS = {
            "hotel", "hotels", "stay", "resort", "villa",
            "price", "budget", "luxury", "rating", "address",
            "amenities", "location", "near", "in"
        }

        is_conversational = (
            any(k in q_lower for k in CONVERSATIONAL_KEYWORDS)
            and not any(d in q_lower for d in DOMAIN_KEYWORDS)
            and len(q_lower.split()) <= 8
        )

        if is_conversational:
            greeting_answer = (
                "Hey! 👋 I'm Anvi, I can help you with hotel searches, place details, "
                "and travel-related questions based on our available data.\n\n"
                "Just tell me what you're looking for 🙂"
            )

            await save_message(app_user_id, "assistant", greeting_answer)

            return {
                "answer": greeting_answer,
                "cards": []
            }

        # 2️⃣ Extract intent
        intent = extract_intent(query)
        category_keyword = intent["category"]
        print("[DEBUG] Stored reply in PostgreSQL memory")


        # 2.5️⃣ Entity + Attribute Bypass (before RAG/LLM)
        if intent.get("type") == "entity_lookup":
            detected_attribute = detect_attribute(query)

            # ✅ Existing: entity + attribute bypass (NO LLM)
            if detected_attribute:
                entity_name = intent.get("entity_name", "")
                if entity_name:
                    print(f"[DEBUG] Entity lookup: {entity_name} | attribute: {detected_attribute} | session={session_id}")
                    entity_data = await resolve_entity(entity_name, intent, token=token)

                    if entity_data:
                        value = entity_data.get(detected_attribute)
                        answer = format_attribute_answer(entity_data, detected_attribute, value)

                        # Store assistant reply in memory
                        await save_message(app_user_id, "assistant", answer)
                        mem_size = 0
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
                    entity_data = await resolve_entity(entity_name, intent, token=token)

                    if entity_data:
                        # Fetch memory for this app_user_id
                        history = await get_recent_messages(app_user_id)
                        memory = "\n".join([f"{m['role']}: {m['content']}" for m in history])
                        mem_size = len(history)
                        print(f"[DEBUG] Memory size={mem_size} | session={session_id}")

                        amenities = entity_data.get("amenities") or []
                        if isinstance(amenities, list) and amenities:
                            amenities_str = ", ".join(amenities[:12])
                        else:
                            amenities_str = "Not provided"

                        description = entity_data.get("description") or "Not provided"
                        if description and len(description) > 200:
                            description = description[:200].rstrip() + "..."

                        # Minimal single-entity context (no multi-entity fetch)
                        entity_context = (
                            "[1]\n"
                            f"Name: {entity_data.get('name') or 'Unknown'}\n"
                            f"Rating: {entity_data.get('rating') or 'Not provided'}\n"
                            f"Address: {entity_data.get('address') or 'Not provided'}\n"
                            f"Amenities: {amenities_str}\n"
                            f"Description: {description}\n"
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
                        # Match structure of generic search cards
                        card = {
                            "title": entity_data.get("name"),
                            "subtitle": entity_data.get("area_name") or entity_data.get("zone_name") or "",
                            "rating": entity_data.get("rating"),
                            "address": entity_data.get("address"),
                            "description": entity_data.get("description") or "",
                            "image": entity_data.get("image_url")
                        }

                        # Store assistant reply in memory
                        await save_message(app_user_id, "assistant", answer)
                        mem_size = 0
                        print(f"[DEBUG] Entity-only response. Memory size={mem_size} | session={session_id}")

                        return {
                            "answer": answer,
                            "cards": [card]
                        }
                    # If entity not found → fall through to existing generic behavior

        # 2.6️⃣ OVERRIDE: Catch entity queries missed by intent detection
        # Fixes cases like "hotel vaishali in nashik" that fall into generic search
        # This runs ONLY if intent type is NOT already "entity_lookup"
        if intent.get("type") != "entity_lookup":
            # Extract potential entity name from query (simple token-based)
            # Skip common stopwords and location words
            q_lower = query.lower()
            stopwords = {
                "what", "is", "the", "of", "tell", "me", "about", "in", "at", "near",
                "rating", "price", "address", "amenities", "phone", "location", "where",
                "map", "directions", "hotel", "hotels", "does", "do", "have", "has",
                "a", "an", "what's", "show", "find", "something", "nashik", "budget",
                "luxury", "family", "couple", "pool", "wifi", "kitchen", "food"
            }
            
            tokens = q_lower.split()
            entity_tokens = [t for t in tokens if t not in stopwords]
            
            if entity_tokens:
                potential_entity_name = " ".join(entity_tokens)
                
                # Attempt entity resolution
                entity_data = await resolve_entity(potential_entity_name, intent, token=token)
                
                if entity_data:
                    # Check if normalized name is NOT generic
                    GENERIC_NAMES = {"hotel", "hotels", "resort", "villa"}
                    entity_normalized = normalize_name(potential_entity_name)
                    
                    if entity_normalized and entity_normalized not in GENERIC_NAMES:
                        # ✅ OVERRIDE: Treat as entity-only lookup
                        print(f"[DEBUG] OVERRIDE: Entity resolved '{potential_entity_name}' → treating as entity-only | session={session_id}")
                        
                        # Fetch memory for this app_user_id
                        history = await get_recent_messages(app_user_id)
                        memory = "\n".join([f"{m['role']}: {m['content']}" for m in history])
                        mem_size = len(history)
                        print(f"[DEBUG] Memory size={mem_size} | session={session_id}")
                        
                        amenities = entity_data.get("amenities") or []
                        if isinstance(amenities, list) and amenities:
                            amenities_str = ", ".join(amenities[:12])
                        else:
                            amenities_str = "Not provided"
                        
                        description = entity_data.get("description") or "Not provided"
                        if description and len(description) > 200:
                            description = description[:200].rstrip() + "..."
                        
                        # Minimal single-entity context (no multi-entity fetch)
                        entity_context = (
                            "[1]\n"
                            f"Name: {entity_data.get('name') or 'Unknown'}\n"
                            f"Rating: {entity_data.get('rating') or 'Not provided'}\n"
                            f"Address: {entity_data.get('address') or 'Not provided'}\n"
                            f"Amenities: {amenities_str}\n"
                            f"Description: {description}\n"
                            "----"
                        )
                        
                        summary_query = (
                            f"Tell me something about {entity_data.get('name') or potential_entity_name}. "
                            "Write a short, factual 2–3 sentence summary using ONLY the provided context."
                        )
                        
                        answer = await answer_with_ai(
                            query=summary_query,
                            context=entity_context,
                            intent=intent,
                            memory=memory
                        )
                        
                        # Single card (do not fetch multiple hotels)
                        # Match structure of generic search cards
                        card = {
                            "title": entity_data.get("name"),
                            "subtitle": entity_data.get("area_name") or entity_data.get("zone_name") or "",
                            "rating": entity_data.get("rating"),
                            "address": entity_data.get("address"),
                            "description": entity_data.get("description") or "",
                            "image": entity_data.get("image_url")
                        }
                        
                        # Store assistant reply in memory
                        await save_message(app_user_id, "assistant", answer)
                        mem_size = 0
                        print(f"[DEBUG] OVERRIDE: Entity-only response. Memory size={mem_size} | session={session_id}")
                        
                        return {
                            "answer": answer,
                            "cards": [card]
                        }
                    # If normalized name is generic → fall through to existing logic

        # 3️⃣ Build RAG context
        context = await get_rag_context(category_keyword, session_id, intent)
        if context:
            print(
                f"[DEBUG] RAG context built (len={len(context)}) | session={session_id}"
            )
        else:
            print(f"[DEBUG] RAG context EMPTY | session={session_id}")

        # 4️⃣ Fetch memory
        history = await get_recent_messages(app_user_id)
        memory = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        mem_size = len(history)
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

        # 6️⃣ Store ASSISTANT reply in memory
        await save_message(app_user_id, "assistant", answer)
        mem_size = 0
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
