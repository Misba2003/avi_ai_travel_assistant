import os
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from jose import jwt, JWTError

from services.intent_service import extract_intent, detect_attribute
from services.data_service import resolve_entity, format_attribute_answer
from services.memory_service import save_message, get_recent_messages
from services.rag_service import get_rag_items
from services.llm_service import answer_with_ai

app = FastAPI()
JWT_ALGORITHM = "HS256"


# ---------------- HEALTH ----------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------- REQUEST ----------------
class AskRequest(BaseModel):
    query: str
    session_id: str | None = None


# Deterministic no-data response (dataset-only; never suggest alternatives).
NO_DATA_MSG = "No matching data found in our listings."

# ---------------- DOMAIN KEYWORDS (HARD FILTER) ----------------
# Must align with intent_service resolver categories. Used for item-text matching.
# STRICT: one domain per query. MIXED: only for exploratory (fun/activities).
DOMAIN_KEYWORDS = {
    "hotel": {"hotel", "hotels", "stay", "lodging", "accommodation"},
    "resort": {"resort", "resorts"},
    "villa": {"villa", "villas"},
    "restaurant": {"restaurant", "restaurants", "cafe", "cafes", "food", "dining"},
    "hospital": {"hospital", "hospitals", "clinic", "medical", "healthcare"},
    "ashram": {"ashram", "ashrams"},
    "theater": {"theater", "theatre", "theaters", "theatres", "cinema", "movie hall", "movie", "movies"},
    "office": {"office", "offices", "coworking", "workspace"},
    "religious": {"temple", "temples", "mandir", "mandirs", "mosque", "church", "religious", "worship"},
    "wine": {"wine", "vineyard", "vineyards", "winery", "wineries"},
    "treks": {"trek", "treks", "trekking", "hiking", "trail", "trails"},
    "adventure": {"adventure", "one-day trip", "day trip", "day trips"},
    "museum": {"museum", "museums"},
    "events": {"event", "events", "festival", "festivals"},
    "picnic": {"picnic", "picnic spot", "picnic spots"},
    "tours": {"tour", "tours", "guided tour", "guided tours", "sightseeing"},
}

# Exploratory: only these categories. Never hospital, office, hotel, resort, villa.
ALLOWED_EXPLORATORY = {
    "theater", "theaters", "museum", "museums",
    "treks", "trek", "picnic", "picnics", "events", "event",
    "tours", "tour", "restaurant", "restaurants",
    "activities", "activity", "adventure", "adventures",
    "wildlife", "wine",
}


def extract_requested_domains(query: str) -> set[str]:
    q = query.lower()
    found = set()
    for domain, words in DOMAIN_KEYWORDS.items():
        if any(w in q for w in words):
            found.add(domain)
    return found


def _first_requested_domain(query: str) -> str | None:
    """First matched domain only (dict order). Used when no resolved_domain provided."""
    q = query.lower()
    for domain, words in DOMAIN_KEYWORDS.items():
        if any(w in q for w in words):
            return domain
    return None


def filter_by_requested_domain(
    items: list[dict], query: str, exploratory: bool = False, resolved_domain: str | None = None
) -> list[dict]:
    """
    HARD deterministic filter. Uses resolved_domain (from keyword resolver) when provided.
    - STRICT: one domain per query (no mixing for hotels, hospitals, offices, etc.).
    - MIXED: only when exploratory=True; keep items matching any MIXED domain.
    """
    if exploratory:
        filtered = []
        for item in items:
            item_cat = (item.get("category") or "").lower().strip()
            if item_cat in ALLOWED_EXPLORATORY:
                filtered.append(item)
        return filtered

    domain = resolved_domain if resolved_domain and resolved_domain in DOMAIN_KEYWORDS else _first_requested_domain(query)
    if domain is None:
        return items

    words = DOMAIN_KEYWORDS[domain]
    filtered = []
    for item in items:
        text = (
            f"{item.get('vendor_name','')} "
            f"{item.get('title','')} "
            f"{item.get('description','')}"
        ).lower()
        if any(w in text for w in words):
            filtered.append(item)

    return filtered


# ---------------- INTRO (FIXED) ----------------
GREETING_WORDS = ("hi", "hello", "hey")

# Self-introduction patterns: "im X", "i'm X", "i am X", "my name is X"
_NAME_PATTERNS = [
    re.compile(r"\bi'?m\s+(.+)", re.I),
    re.compile(r"\bi\s+am\s+(.+)", re.I),
    re.compile(r"\bmy\s+name\s+is\s+(.+)", re.I),
]


def _has_greeting(query: str) -> bool:
    q = query.lower().strip()
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in GREETING_WORDS)


def _extract_introduced_name(query: str) -> str | None:
    """Extract name X from self-intro patterns. Deterministic, no LLM."""
    q = query.strip()
    for pat in _NAME_PATTERNS:
        m = pat.search(q)
        if m:
            name = m.group(1).strip().rstrip(".,!?")
            if name and len(name) <= 80:
                return name
    return None


INTRO_PHRASES = {
    "hi", "hello", "hey",
    "who are you",
    "what can you do",
    "tell me about yourself",
    "introduce yourself",
    "how can you help",
}


def is_intro_query(query: str) -> bool:
    q = query.lower().strip()
    for p in INTRO_PHRASES:
        if " " in p:
            if p in q:
                return True
        else:
            # WORD BOUNDARY FIX (hi ≠ nashik)
            if re.search(rf"\b{re.escape(p)}\b", q):
                return True
    return False


MEMORY_QUESTION_PHRASES = (
    "what did i ask", "what did i say", "previous question", "last question",
    "my last message", "what was my question", "earlier question",
    "what did i ask earlier", "what did i say earlier",
)


def _is_memory_question(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in MEMORY_QUESTION_PHRASES)


FOLLOW_UP_WORDS = ("yes", "more", "continue")


def _is_follow_up(query: str) -> bool:
    return query.strip().lower() in FOLLOW_UP_WORDS


def _get_previous_user_message(messages: list) -> str | None:
    """Last user message before the current one (current is last in list)."""
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    return user_contents[-2] if len(user_contents) >= 2 else None


# ---------------- MAIN ----------------
@app.post("/ask")
async def ask_ai(req: AskRequest, authorization: str = Header(None)):

    # ---------- AUTH ----------
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
    await save_message(user_id, "user", query)

    # ---------- INTRO: greeting + self-introduction (name) → acknowledge name, no RAG ----------
    intro_name = _extract_introduced_name(query)
    if _has_greeting(query) and intro_name:
        answer = (
            f"Hi {intro_name}! Nice to meet you. I'm Anvi AI, your Nashik travel assistant. "
            "How can I help you with your travel plans today?"
        )
        await save_message(user_id, "assistant", answer)
        return {"answer": answer, "cards": []}

    # ---------- INTRO: greeting only or other intro phrases ----------
    if is_intro_query(query):
        answer = (
            "Hi! I’m Anvi AI, a Nashik-based travel assistant. "
            "You can ask me to show, find, or tell you about hotels, resorts, villas, "
            "restaurants, theaters, ashrams, and other city activities in Nashik."
        )
        await save_message(user_id, "assistant", answer)
        return {"answer": answer, "cards": []}

    # ---------- MEMORY-QUESTION GUARD (before intent) ----------
    if _is_memory_question(query):
        messages = await get_recent_messages(user_id, limit=20)
        prev_user = _get_previous_user_message(messages)
        if prev_user:
            answer = f'You asked: "{prev_user}"'
        else:
            answer = "I don't have a previous question from you in this conversation."
        await save_message(user_id, "assistant", answer)
        return {"answer": answer, "cards": []}

    # ---------- FOLLOW-UP GUARD: reuse previous query/intent (before intent) ----------
    effective_query = query
    if _is_follow_up(query):
        messages = await get_recent_messages(user_id, limit=20)
        prev_user = _get_previous_user_message(messages)
        if prev_user:
            effective_query = prev_user

    # ---------- INTENT ----------
    intent = extract_intent(effective_query) or {}

    # =========================================================
    # GLOBAL ENTITY + ATTRIBUTE BYPASS (runs for ALL queries)
    # =========================================================
    detected_attribute = detect_attribute(query)

    # Try resolving entity regardless of intent type
    potential_entity_name = intent.get("entity_name") or intent.get("entity") or query

    entity_data = await resolve_entity(potential_entity_name, intent, token=token)

    if entity_data and detected_attribute:
        print(f"[DEBUG] GLOBAL entity+attribute → {entity_data.get('name')} | attr={detected_attribute}")

        value = entity_data.get(detected_attribute)
        answer = format_attribute_answer(entity_data, detected_attribute, value)

        await save_message(user_id, "assistant", answer)
        print(f"[DEBUG] GLOBAL attribute response stored")

        return {
            "answer": answer,
            "cards": []
        }

    # --- ENTITY ATTRIBUTE SHORT-CIRCUIT ---
    if intent.get("type") == "entity_attribute":
        entity_name = intent.get("entity_name")
        attributes = intent.get("attributes") or []

        entity = await resolve_entity(entity_name, intent, token=token)

        if not entity:
            await save_message(user_id, "assistant", f"I couldn't find {entity_name} in our listings.")
            return {
                "answer": f"I couldn't find {entity_name} in our listings.",
                "cards": []
            }

        answer_parts = [
            format_attribute_answer(entity, attr, entity.get(attr))
            for attr in attributes
        ]
        answer = " ".join(answer_parts) if answer_parts else "No attributes requested."

        await save_message(user_id, "assistant", answer)
        return {
            "answer": answer,
            "cards": [entity]
        }
    # --- END ENTITY ATTRIBUTE SHORT-CIRCUIT ---

    # Align RAG domain filter with intent: use same key as main.py DOMAIN_KEYWORDS.
    if intent.get("search_domain"):
        intent["category"] = intent["search_domain"]
    intent_type = intent.get("type")

    # ===================================================
    # ENTITY LOOKUP (SINGLE PLACE FLOW)
    # ===================================================
    if intent_type == "entity_lookup":
        entity_name = intent.get("entity_name")

        items = await get_rag_items(effective_query, intent)

        entity_data = items[0] if len(items) == 1 else None

        if not entity_data and entity_name:
            entity_data = await resolve_entity(entity_name, intent, token=token)

        if entity_data:
            attr = detect_attribute(query)
            if attr:
                answer = format_attribute_answer(
                    entity_data, attr, entity_data.get(attr)
                )
                await save_message(user_id, "assistant", answer)
                return {"answer": answer, "cards": []}

            history = await get_recent_messages(user_id)
            answer = await answer_with_ai(
                query=query,
                context=entity_data.get("description", ""),
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

        await save_message(user_id, "assistant", NO_DATA_MSG)
        return {"answer": NO_DATA_MSG, "cards": []}

    # ===================================================
    # GENERAL SEARCH (LIST FLOW)
    # ===================================================
    items = await get_rag_items(effective_query, intent)

    # HARD STOP if amenity requested but no match
    if intent.get("must_have") and not items:
        await save_message(user_id, "assistant", NO_DATA_MSG)
        return {"answer": NO_DATA_MSG, "cards": []}

    # Build RAG context ONLY from these filtered items
    rag_lines = []
    for idx, i in enumerate(items[:8], 1):
        name = i.get("vendor_name") or i.get("name") or "Unknown"
        area = i.get("area_name") or ""
        rating = i.get("star_rating") or ""
        address = i.get("address") or ""
        desc = (i.get("short_description") or i.get("description") or "")[:200]

        rag_lines.append(
            f"[{idx}]\n"
            f"Name: {name}\n"
            f"Area: {area}\n"
            f"Rating: {rating}\n"
            f"Address: {address}\n"
            f"Description: {desc}\n----"
        )

    rag_context = "\n".join(rag_lines).strip()

    history = await get_recent_messages(user_id)

    answer = await answer_with_ai(
        query=effective_query,
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
