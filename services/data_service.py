# services/data_service.py

import os
from typing import Any, Dict, List
from fastapi import Query
import httpx
from dotenv import load_dotenv

from utils.image_utils import build_image_url

load_dotenv()

API_TOKEN = os.getenv("NASHIK_API_TOKEN", "").strip()
BASE_URL = "https://nashikguide.sapphiredigital.agency/api/search/"


def normalize_name(name: str) -> str:
    """
    Normalize a name for matching by:
    - lowercasing
    - removing the words "hotel" and "the"
    - stripping extra spaces
    """
    if not name:
        return ""
    normalized = name.lower().strip()
    # Remove stopwords as whole tokens
    tokens = normalized.split()
    tokens = [t for t in tokens if t not in ("hotel", "the")]
    return " ".join(tokens).strip()


def _normalize_name_for_matching(name: str) -> str:
    """
    Normalize a name for matching by:
    - lowercasing
    - removing the word "hotel"
    - stripping extra spaces
    """
    return normalize_name(name)


def find_exact_hotel(items, query_name: str):
    """
    Find hotel using normalized token-based comparison.
    Matches using containment after normalizing both query and hotel names.
    """
    query_normalized = _normalize_name_for_matching(query_name)
    
    if not query_normalized:
        return None

    for item in items:
        # Try vendor_name first
        vendor_name = item.get("vendor_name", "")
        if vendor_name:
            vendor_normalized = _normalize_name_for_matching(vendor_name)
            if query_normalized in vendor_normalized or vendor_normalized in query_normalized:
                return item
        
        # Try name field
        name = item.get("name", "")
        if name:
            name_normalized = _normalize_name_for_matching(name)
            if query_normalized in name_normalized or name_normalized in query_normalized:
                return item

    return None


def normalize_hotel_entity(hotel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize hotel entity data into a consistent structure.
    """
    amenities_list = []
    if isinstance(hotel.get("amenities_gallery"), list):
        for a in hotel.get("amenities_gallery", []):
            if isinstance(a, dict) and a.get("amenity"):
                amenities_list.append(a["amenity"])
    
    # Check amenities_gallery for wifi, pool, bonfire
    amenities_lower = [a.lower() for a in amenities_list]
    has_wifi = any("wifi" in a or "wi-fi" in a for a in amenities_lower)
    has_pool = any("pool" in a for a in amenities_lower)
    has_bonfire = any("bonfire" in a for a in amenities_lower)
    
    return {
        "name": hotel.get("name") or hotel.get("vendor_name"),
        "rating": hotel.get("star_rating"),
        "address": hotel.get("address"),
        "phone": hotel.get("phone"),
        "email": hotel.get("email"),
        "website": hotel.get("website"),
        "pet_friendly": hotel.get("pet_friendly") == "Y" or hotel.get("pet_friendly") == True,
        "parking": hotel.get("parking_available") == "Y" or hotel.get("parking_available") == True,
        "air_conditioned": hotel.get("air_conditioned") == "Y" or hotel.get("air_conditioned") == True,
        "food_available": hotel.get("food_available") == "Y" or hotel.get("food_available") == True,
        "price_from": hotel.get("price_from"),
        "price_unit": hotel.get("price_unit"),
        "map": hotel.get("google_location"),
        "amenities": amenities_list,
        "wifi": has_wifi,
        "pool": has_pool,
        "bonfire": has_bonfire,
        "google_location": hotel.get("google_location"),
        "kitchen_available": hotel.get("kitchen_available") == "Y" or hotel.get("kitchen_available") == True,
        "taxes_included": hotel.get("taxes_included") == "Y" or hotel.get("taxes_included") == True,
        "cancellation": hotel.get("cancellation"),
        # Card fields (for entity-only queries)
        "image_url": hotel.get("image_url"),
        "area_name": hotel.get("area_name"),
        "zone_name": hotel.get("zone_name"),
        "description": hotel.get("description") or hotel.get("short_description"),
        # Backend-only identifiers for internal navigation/filtering
        "table_id": hotel.get("table_id"),
        "category_id": hotel.get("category_id"),
    }


async def resolve_entity(
    entity_name: str,
    intent: Dict[str, Any],
    token: str | None = None,
) -> Dict[str, Any] | None:
    """
    Resolve a single entity (hotel) by name from the API.
    Returns normalized entity data or None if not found.
    
    This function performs deterministic entity lookup by:
    1. Fetching a large result set (limit=200) from the API
    2. Performing name-based matching over ALL fetched items
    3. NOT relying on ranking or _score for entity resolution
    """
    # Fetch items directly from API without ranking
    # This ensures we check ALL items, not just top-ranked results
    params = {
        "query": entity_name,
        "page": 1,
        "limit": 200,
    }

    # Prefer caller-provided Bearer token; fall back to .env token
    effective_token = (token or "").strip() or API_TOKEN

    headers = {
        "Authorization": f"Bearer {effective_token}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as e:
        print("[ERROR] resolve_entity API exception:", e)
        return None

    # Extract raw items
    raw_items: List[Dict[str, Any]] = []
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("data"), dict)
        and isinstance(payload["data"].get("search_data"), list)
    ):
        raw_items = payload["data"]["search_data"]

    if not raw_items:
        return None

    # Normalize items (add image_url, but skip ranking/scoring)
    items: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        image_path = None
        if item.get("thumbnail_image"):
            image_path = item["thumbnail_image"]
        elif isinstance(item.get("gallery_images"), list) and item["gallery_images"]:
            image_path = item["gallery_images"][0]

        item["image_url"] = build_image_url(image_path)
        items.append(item)

    # ----------------------------------------
    # Deterministic, guarded entity resolution
    # ----------------------------------------
    GENERIC_NAMES = {"hotel", "hotels", "resort", "villa"}

    entity_normalized = normalize_name(entity_name)
    if not entity_normalized:
        return None

    # PASS 1 — exact match ONLY (on normalized names)
    for item in items:
        vendor_name = item.get("vendor_name", "")
        if vendor_name:
            vendor_normalized = normalize_name(vendor_name)
            if vendor_normalized and entity_normalized == vendor_normalized:
                return normalize_hotel_entity(item)

        name = item.get("name", "")
        if name:
            name_normalized = normalize_name(name)
            if name_normalized and entity_normalized == name_normalized:
                return normalize_hotel_entity(item)

    # PASS 2 — fuzzy/contains match, guarded against generic names
    for item in items:
        vendor_name = item.get("vendor_name", "")
        if vendor_name:
            vendor_normalized = normalize_name(vendor_name)
            if vendor_normalized and vendor_normalized not in GENERIC_NAMES:
                if (
                    entity_normalized in vendor_normalized
                    or vendor_normalized in entity_normalized
                ):
                    return normalize_hotel_entity(item)

        name = item.get("name", "")
        if name:
            name_normalized = normalize_name(name)
            if name_normalized and name_normalized not in GENERIC_NAMES:
                if (
                    entity_normalized in name_normalized
                    or name_normalized in entity_normalized
                ):
                    return normalize_hotel_entity(item)

    # No match found after checking all items
    return None


def format_attribute_answer(entity_data: Dict[str, Any], attribute: str, value: Any) -> str:
    """
    Format a direct factual answer for an entity attribute query.
    """
    entity_name = entity_data.get("name", "This hotel")
    
    # Special handling for price (uses price_from and price_unit, not a single "price" field)
    if attribute == "price":
        price_from = entity_data.get("price_from")
        price_unit = entity_data.get("price_unit", "")
        if price_from:
            return f"{entity_name}'s price starts from {price_from} {price_unit}.".strip()
        return f"{entity_name} does not have price information available."
    
    if value is None or value == "":
        return f"{entity_name} does not have {attribute} information available."
    
    if attribute == "rating":
        try:
            value = float(value)
            return f"{entity_name} has a {value}-star rating."
        except Exception:
            return f"{entity_name} has a rating of {value}."
    
    if attribute == "address":
        return f"{entity_name} is located at {value}."
    
    if attribute == "phone":
        return f"{entity_name}'s phone number is {value}."
    
    if attribute == "amenities":
        if isinstance(value, list) and value:
            amenities_str = ", ".join(value[:5])  # Limit to first 5
            if len(value) > 5:
                amenities_str += f" and {len(value) - 5} more"
            return f"{entity_name} offers: {amenities_str}."
        return f"{entity_name} does not have amenities information available."
    
    if attribute == "parking":
        return f"{entity_name} {'has' if value else 'does not have'} parking available."
    
    if attribute == "pet_friendly":
        return f"{entity_name} is {'pet-friendly' if value else 'not pet-friendly'}."
    
    if attribute == "map":
        if value:
            return f"{entity_name}'s location: {value}"
        return f"{entity_name} does not have location/map information available."
    
    if attribute == "wifi":
        return f"{entity_name} {'has' if value else 'does not have'} WiFi available."
    
    if attribute == "pool":
        return f"{entity_name} {'has' if value else 'does not have'} a pool."
    
    if attribute == "bonfire":
        return f"{entity_name} {'has' if value else 'does not have'} bonfire facilities."
    
    if attribute == "google_location":
        if value:
            return f"{entity_name} is located at {value}."
        return f"{entity_name} does not have location information available."
    
    if attribute == "website":
        if value:
            return f"{entity_name}'s website is {value}."
        return f"{entity_name} does not have a website listed."
    
    if attribute == "kitchen_available":
        return f"{entity_name} {'has' if value else 'does not have'} a kitchen available."
    
    if attribute == "food_available":
        return f"{entity_name} {'has' if value else 'does not have'} food available."
    
    if attribute == "taxes_included":
        return f"{entity_name} {'includes' if value else 'does not include'} taxes in the price."
    
    if attribute == "price_unit":
        if value:
            return f"{entity_name}'s price is per {value}."
        return f"{entity_name} does not have price unit information available."
    
    if attribute == "cancellation":
        if value:
            return f"{entity_name}'s cancellation policy: {value}."
        return f"{entity_name} does not have cancellation policy information available."

    if attribute == "air_conditioned":
       if value:
        return f"{entity_name} has air-conditioned rooms."
       return f"{entity_name} does not have  air-conditioned rooms."


def canonical_category(raw_category: str) -> str:
    if not raw_category:
        return ""

    c = raw_category.lower().strip()

    if "hotel" in c:
        return "hotel"
    if "resort" in c:
        return "resort"
    if "villa" in c:
        return "villa"
    if "restaurant" in c or "cafe" in c:
        return "restaurant"
    if "medical" in c or "hospital" in c:
        return "hospital"
    if "office" in c:
        return "office"
    # Theaters
    if "theater" in c or "theatre" in c or "theaters" in c or "theatres" in c:
        return "theater"
    if "museum" in c:
        return "museum"
    if "religious" in c or "temple" in c or "mandir" in c or "ashram" in c:
        return "religious"
    if "trek" in c:
        return "treks"
    if "adventure" in c or "one-day" in c:
        return "adventure"
    if "wildlife" in c or "nature" in c:
        return "wildlife"
    if "picnic" in c:
        return "picnic"
    if "wine" in c:
        return "wine"
    if "shopping" in c:
        return "shopping"

    return c


def score_item(item: Dict[str, Any], intent: Dict[str, Any]) -> int:
    """
    Simple keyword-based scoring
    """
    score = 0
    text = (
        f"{item.get('sub_category','')} "
        f"{item.get('category','')} "
        f"{item.get('description','')}"
    ).lower()

    for kw in intent.get("keywords", []):
        if kw in text:
            score += 1

    return score


async def search_api(
    query: str,
    intent: Dict[str, Any],
    page: int = 1,
    limit: int = 30,
    token: str | None = None,
) -> List[Dict[str, Any]]:

    search_domain = intent.get("search_domain") or query

    params = {
        "query": search_domain,
        "page": 1,
        "limit": 200,
    }

    # Prefer caller-provided Bearer token; fall back to .env token
    effective_token = (token or "").strip() or API_TOKEN

    headers = {
        "Authorization": f"Bearer {effective_token}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as e:
        print("[ERROR] search_api exception:", e)
        return []

    # -------------------------------
    # Extract raw items
    # -------------------------------
    raw_items: List[Dict[str, Any]] = []

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("data"), dict)
        and isinstance(payload["data"].get("search_data"), list)
    ):
        raw_items = payload["data"]["search_data"]

    print("[DEBUG] RAW API item count:", len(raw_items))

    # -------------------------------
    # Normalize + image
    # -------------------------------
    normalized: List[Dict[str, Any]] = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        image_path = None

        if item.get("thumbnail_image"):
            image_path = item["thumbnail_image"]
        elif isinstance(item.get("gallery_images"), list) and item["gallery_images"]:
            image_path = item["gallery_images"][0]

        item["image_url"] = build_image_url(image_path)
        item["normalized_category"] = canonical_category(item.get("category"))
        normalized.append(item)

    # -------------------------------
    # Intent-based ranking
    # -------------------------------
    for item in normalized:
        item["_score"] = score_item(item, intent)

    # If keyword matches exist, rank them
    matched = [i for i in normalized if i["_score"] > 0]

    if matched:
        matched.sort(key=lambda x: x["_score"], reverse=True)
        return matched[:limit]

    # If no keyword match, return all normalized items
    return normalized[:limit]
