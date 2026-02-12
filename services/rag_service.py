# services/rag_service.py

import asyncio
from typing import List, Dict

from services.data_service import search_api

MAX_RESULTS = 8  # keep your existing window

# Exploratory: only these categories. Never hotel, hospital, office, resort, villa.
ALLOWED_EXPLORATORY = {
    "museum",
    "theater",
    "treks",
    "picnic",
    "events",
    "adventure",
    "wildlife",
    "restaurant",
    "shopping",
    "wine",
}


def normalize_category(raw_category: str) -> str:
    if not raw_category:
        return ""

    c = raw_category.lower().strip()

    # Hotels
    if "hotel" in c:
        return "hotel"
    if "resort" in c:
        return "resort"
    if "villa" in c:
        return "villa"

    # Restaurants / Cafes
    if "restaurant" in c or "cafe" in c:
        return "restaurant"

    # Hospitals
    if "medical" in c or "hospital" in c:
        return "hospital"

    # Offices
    if "office" in c:
        return "office"

    # Theaters
    if "theater" in c:
        return "theater"

    # Museums
    if "museum" in c:
        return "museum"

    # Religious
    if "religious" in c or "mandir" in c or "temple" in c or "ashram" in c:
        return "religious"

    # Adventure / Treks / One Day
    if "trek" in c:
        return "treks"
    if "adventure" in c or "one-day" in c:
        return "adventure"

    # Wildlife
    if "wildlife" in c or "nature" in c:
        return "wildlife"

    # Picnic
    if "picnic" in c:
        return "picnic"

    # Wine
    if "wine" in c:
        return "wine"

    # Shopping
    if "shopping" in c:
        return "shopping"

    return c


def _item_category(item: Dict) -> str:
    """Item category from ingestion (normalized_category)."""
    return item.get("normalized_category", "")


def _matches_intent_category(item: Dict, intent: Dict) -> bool:
    """
    Strict category filter when not exploratory.
    Exploratory: allow all items with non-empty normalized_category.
    """
    item_category = (item.get("normalized_category") or "").strip()

    if intent.get("exploratory"):
        return bool(item_category)

    intent_category = (intent.get("search_domain") or "").lower().strip()
    if intent_category and item_category != intent_category:
        return False
    return True


async def _format_item(item: Dict, index: int) -> str:
    name = item.get("vendor_name") or item.get("name") or "Unknown"
    category = item.get("category") or item.get("type") or ""
    area = item.get("area_name") or item.get("zone_name") or item.get("area") or ""
    rating = item.get("rating") or item.get("star_rating") or ""
    address = (
        item.get("address")
        or item.get("location")
        or item.get("area_name")
        or ""
    )
    desc = item.get("short_description") or item.get("description") or ""

    if desc and len(desc) > 200:
        desc = desc[:200].rstrip() + "..."

    return (
        f"[{index}]\n"
        f"Name: {name}\n"
        f"Category: {category}\n"
        f"Area: {area}\n"
        f"Rating: {rating}\n"
        f"Address: {address}\n"
        f"Description: {desc}\n"
        f"----"
    )


async def get_rag_context(keyword: str, session_id: str, intent: Dict) -> str:
    """
    RAG context builder with STRICT domain filtering.
    LLM does NOT decide filtering.
    """

    items = await search_api(keyword, intent, limit=30)

    if not items:
        return ""

    # ---------------- CATEGORY FILTER ----------------
    if intent.get("exploratory"):
        items = [i for i in items if (i.get("normalized_category") or "").strip()]
    else:
        intent_category = (intent.get("search_domain") or "").lower().strip()
        if intent_category:
            items = [i for i in items if (i.get("normalized_category") or "").strip() == intent_category]

    if not items:
        return ""

    filtered = [i for i in items if _matches_intent_category(i, intent)]

    if not filtered:
        return ""

    selected = filtered[:MAX_RESULTS]

    formatted = await asyncio.gather(
        *[_format_item(item, idx + 1) for idx, item in enumerate(selected)]
    )

    return "\n".join(formatted).strip()


async def get_rag_items(keyword: str, intent: Dict) -> List[Dict]:
    items = await search_api(keyword, intent, limit=30)

    if intent.get("exploratory"):
        items = [i for i in items if (i.get("normalized_category") or "").strip()]
    else:
        intent_category = (intent.get("search_domain") or "").lower().strip()
        if intent_category:
            items = [i for i in items if (i.get("normalized_category") or "").strip() == intent_category]

    return [i for i in items if _matches_intent_category(i, intent)]