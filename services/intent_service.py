from typing import Dict, Any, List, Optional
import re

# -----------------------------
# Attribute keywords (KEEP AS-IS)
# -----------------------------
ATTRIBUTE_KEYWORDS = {
    "rating": ["rating", "stars", "star"],
    "address": ["address", "where"],
    "phone": ["phone", "contact", "number"],
    "amenities": ["amenities", "facilities", "features"],
    "parking": ["parking"],
    "pet_friendly": ["pet", "pets", "pet-friendly"],
    "price": ["price", "cost", "tariff", "rate", "rates"],
    "map": ["map", "directions", "location"],
    "vendor_name": ["vendor name", "vendor"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "pool": ["pool", "swimming"],
    "bonfire": ["bonfire"],
    "google_location": ["google location"],
    "website": ["website", "site", "url"],
    "kitchen_available": ["kitchen"],
    "food_available": ["food"],
    "taxes_included": ["tax", "taxes", "tax included"],
    "price_unit": ["price_unit", "unit"],
    "cancellation": ["cancellation", "cancel"],
    "air_conditioned": ["ac", "air conditioned", "air-conditioning"]
}

# -----------------------------
# KEYWORD-TO-DATASET-CATEGORY RESOLVER (fixed mapping, no LLM)
# Longest phrases first so "wine shop" matches before "wine".
# Maps user language to existing dataset categories before intent filtering.
# -----------------------------
PHRASE_TO_CATEGORY = (
    # (phrase_or_word, dataset_category)
    ("wine shop", "wine"),
    ("wine shops", "wine"),
    ("one day trip", "adventure"),
    ("one-day trip", "adventure"),
    ("day trips", "adventure"),
    ("guided tour", "tours"),
    ("guided tours", "tours"),
    ("picnic spot", "picnic"),
    ("picnic spots", "picnic"),
    ("movie theater", "theater"),
    ("movie theatres", "theater"),
    ("religious services", "religious"),
    ("mandir", "religious"),
    ("mandirs", "religious"),
    ("temples", "religious"),
    ("temple", "religious"),
    ("movie", "theater"),
    ("movies", "theater"),
    ("cinema", "theater"),
    ("cafe", "restaurant"),
    ("cafes", "restaurant"),
    ("trekking", "treks"),
    ("trek", "treks"),
    ("treks", "treks"),
    ("vineyard", "wine"),
    ("vineyards", "wine"),
    ("winery", "wine"),
    ("wineries", "wine"),
    ("wine", "wine"),
    ("ashram", "ashram"),
    ("ashrams", "ashram"),
    ("resort", "resort"),
    ("resorts", "resort"),
    ("villa", "villa"),
    ("villas", "villa"),
    ("hotel", "hotel"),
    ("hotels", "hotel"),
    ("restaurant", "restaurant"),
    ("restaurants", "restaurant"),
    ("hospital", "hospital"),
    ("hospitals", "hospital"),
    ("clinic", "hospital"),
    ("medical", "hospital"),
    ("office", "office"),
    ("offices", "office"),
    ("theater", "theater"),
    ("theatre", "theater"),
    ("theatres", "theater"),
    ("museum", "museum"),
    ("museums", "museum"),
    ("event", "events"),
    ("events", "events"),
    ("festival", "events"),
    ("festivals", "events"),
    ("mosque", "religious"),
    ("church", "religious"),
    ("worship", "religious"),
)


def resolve_query_to_category(query: str) -> str | None:
    """Deterministic: map user query to one dataset category. No LLM. Longest phrase first."""
    q = query.lower().strip()
    # Sort by phrase length descending so "wine shop" matches before "wine"
    for phrase, category in sorted(PHRASE_TO_CATEGORY, key=lambda x: -len(x[0])):
        if phrase in q:
            return category
    return None


# Words that are category-only (list queries, not entity lookup).
PURE_CATEGORY_WORDS = {
    "hotel", "hotels", "resort", "resorts", "villa", "villas",
    "restaurant", "restaurants", "cafe", "cafes", "theater", "theatres",
    "hospital", "hospitals", "office", "offices", "ashram", "ashrams",
    "medical", "lodging", "food", "movies", "cinema", "nashik",
}

# -----------------------------
# STOPWORDS (KEEP AS-IS)
# -----------------------------
STOPWORDS = {
    "what", "is", "the", "of", "tell", "me", "about",
    "rating", "price", "address", "amenities", "phone",
    "location", "where", "map", "directions",
    "hotel", "does", "do", "have", "has", "a", "an",
    "what's", "show", "find", "something",
    "wifi", "wi-fi", "internet", "pool", "swimming", "bonfire",
    "website", "site", "url", "kitchen", "food",
    "tax", "taxes", "cancellation", "cancel", "unit"
}

# -----------------------------
# MAIN INTENT PARSER
# -----------------------------
def detect_intent(query: str) -> Dict[str, Any]:
    q = query.lower()

    intent: Dict[str, Any] = {
        "raw_query": query,
        "action": "search",              # show / find / list / tell
        "search_domain": None,            # hotels, resorts, villas, etc.
        "attributes": [],                # rating, amenities, etc.
        "must_have": [],                 # pool, family, luxury, budget
        "entity": None                   # specific name (hotel / place)
    }

    # -----------------------------
    # Action detection (NEW – SAFE)
    # -----------------------------
    if any(w in q for w in ["show", "list", "find", "search"]):
        intent["action"] = "list"
    elif any(w in q for w in ["tell", "about", "details"]):
        intent["action"] = "detail"
    elif any(w in q for w in ["who are you", "what can you do", "about yourself", "hey", "hi"]):
        intent["action"] = "general"

    # -----------------------------
    # Search domain: resolver first (keyword → dataset category), then DOMAIN_KEYWORDS
    # -----------------------------
    resolved = resolve_query_to_category(query)
    if resolved:
        intent["search_domain"] = resolved
    else:
        DOMAIN_KEYWORDS = {
            "hotel": ["hotel", "hotels", "stay", "stays", "lodging", "accommodation"],
            "resort": ["resort", "resorts"],
            "villa": ["villa", "villas"],
            "restaurant": ["restaurant", "restaurants", "cafe", "cafes", "food", "restaurants & cafes", "eating", "dining", "eat"],
            "events": ["event", "events", "festival", "festivals", "festivities", "happenings", "events & festivals"],
            "treks": ["trek", "treks", "hiking", "trekking", "trail", "trails"],
            "activities": ["activity", "activities", "things to do", "city activities", "city activity", "things to do in city"],
            "wine": ["wine", "vineyard", "winery", "wineries"],
            "shopping": ["shopping", "mall", "market", "malls", "markets", "shop", "stores"],
            "religious": ["temple", "mosque", "church", "religious", "religious services", "worship"],
            "wellness": ["ayurveda", "spa", "wellness", "ayurvedic", "massage", "retreat", "wellness center", "ayurveda & wellness"],
            "kids": ["kids", "children", "activities for kids", "kids activities", "children activities", "family activities", "activities for children"],
            "attractions": ["attraction", "attractions", "places to visit", "places to explore", "places to see", "places to discover"],
            "visitors": ["visitor", "visitors", "visitor visit", "visitor visits"],
            "art": ["art", "art gallery", "art galleries", "art museum", "art museums"],
            "museum": ["museum", "museums", "museum visit", "museum visits"],
            "history": ["history", "historical", "historical places", "historical places to visit", "historical places to see", "historical places to explore", "historical places to discover"],
            "hospital": ["healthcare", "hospital", "hospitals", "clinic", "medical", "nursing", "management", "dental", "health", "diagnostic", "pathology", "care center", "iccu", "icu", "selfcare", "emergency", "emergency services", "medical services", "ambulance", "doctor", "pharmacy", "emergency & medical"],
            "ashram": ["ashram", "ashrams", "ashram visit", "ashram visits"],
            "office": ["office", "offices", "office visit", "office visits", "virtual office", "virtual offices", "coworking", "coworking space", "workspace"],
            "theater": ["movie", "movies", "movie theater", "movie theaters", "theater", "theatre", "theaters", "theatres", "cinema"],
            "adventure": ["adventure", "one-day trip", "day trip", "one day trip", "day trips", "adventure activities", "adventure & one-day trips"],
            "wildlife": ["wildlife", "nature", "safari", "national park", "nature spots", "wildlife sanctuary", "wildlife & nature", "nature spots"],
            "picnic": ["picnic", "picnic spot", "picnic spots", "picnic area", "picnic spots"],
            "tours": ["guided tour", "guided tours", "tours", "tour", "sightseeing", "guided tours"],
            "today_happenings": ["today's happenings", "today happenings", "what's on today", "whats on today", "going on today", "today events", "happenings today", "today's happenings", "what is happening today"],
        }
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(k in q for k in keywords):
                intent["search_domain"] = domain
                break

    # -----------------------------
    # Attribute detection (KEEP)
    # -----------------------------
    for attr, keywords in ATTRIBUTE_KEYWORDS.items():
        if any(k in q for k in keywords):
            intent["attributes"].append(attr)

    # -----------------------------
    # Filters (KEEP + WORKING)
    # -----------------------------
    if "pool" in q:
        intent["must_have"].append("pool")

    if "family" in q:
        intent["must_have"].append("family")

    if "couple" in q:
        intent["must_have"].append("couple")

    if "luxury" in q:
        intent["must_have"].append("luxury")

    if "budget" in q or "cheap" in q:
        intent["must_have"].append("budget")

    # -----------------------------
    # Entity extraction (SAFE)
    # -----------------------------
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", q)
    entity_tokens = [t for t in tokens if t not in STOPWORDS]

    if entity_tokens:
        intent["entity"] = " ".join(entity_tokens)

    # -----------------------------
    # Exploratory (mixed-domain discovery)
    # -----------------------------
    exploratory_phrases = [
        "fun activities", "things to do", "what to do", "discover", "explore",
        "experiences", "activities in", "fun in", "something to do",
    ]
    intent["exploratory"] = any(p in q for p in exploratory_phrases)

    # -----------------------------
    # Entity lookup vs list (type + entity_name)
    # -----------------------------
    entity_val = intent.get("entity") or ""
    entity_lower = entity_val.lower().strip()
    entity_words = set(entity_lower.split())
    is_pure_category = (
        len(entity_words) == 1 and entity_lower in PURE_CATEGORY_WORDS
    ) or (entity_words and entity_words <= PURE_CATEGORY_WORDS)
    if (
        intent.get("action") == "detail"
        and entity_val
        and not is_pure_category
    ):
        intent["type"] = "entity_lookup"
        intent["entity_name"] = entity_val.strip()

    return intent
# ----------------------------------
# BACKWARD COMPATIBILITY (DO NOT REMOVE)
# ----------------------------------

def extract_intent(query: str) -> Dict[str, Any]:
    return detect_intent(query)


def detect_attribute(query: str) -> Optional[str]:
    intent = detect_intent(query)
    if intent.get("attributes"):
        return intent["attributes"][0]
    return None
