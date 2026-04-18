"""
OrbitalGuard AI — Narad Chatbot (Upgraded)
===========================================
AI chatbot for querying information about tracked orbital objects.

Supports:
  - NORAD 5-digit ID lookup:  "What is 25544?"
  - Alphanumeric ID lookup:   "Tell me about IE124"
  - Name keyword search:      "What is STARLINK-1234?"
  - Statistics queries:       "How many debris objects are tracked?"
  - Risk queries:             "How many HIGH risk objects are there?"
  - Type queries:             "List all rocket bodies"
  - Help                      "What can you ask?"

All responses draw from the real debris_catalog.json via DebrisCatalog.
"""

import os
import re
import sys

# ─── Path Alignment ──────────────────────────────────────
KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT     = os.path.dirname(KNOWLEDGE_DIR)
if AIML_ROOT not in sys.path:
    sys.path.insert(0, AIML_ROOT)

from knowledge.debris_catalog import DebrisCatalog


# ────────────────────────────────────────────────────────────
HELP_TEXT = """I am **Narad AI** — the OrbitalGuard Intelligence Engine.

I can answer questions about any tracked orbital object. Try:

  🔍  "What is debris 25544?"
  🔍  "Tell me about STARLINK-1234"
  🔍  "What type is IE124?"
  📊  "How many objects are tracked?"
  📊  "How many debris objects are there?"
  📊  "Show stats"
  🛰  "What are rocket bodies?"
  ❓  "Help" / "What can you ask?"
"""

TYPE_INFO = {
    'payload': (
        "A **payload** is an active or inactive satellite launched for a "
        "specific mission (communications, Earth observation, navigation, etc.). "
        "Payloads are the intended operational objects — not all are still active."
    ),
    'debris': (
        "**Space debris** (also called space junk) includes fragmented pieces "
        "from satellite breakups, explosions, or collisions — paint flakes, "
        "metal fragments, and defunct satellites. Even 1cm fragments travel at "
        "~7 km/s and can destroy active satellites."
    ),
    'rocket': (
        "A **rocket body** is the upper stage or booster left in orbit after "
        "a satellite launch. They are large, derelict objects that pose a "
        "significant collision risk and contribute to the Kessler Syndrome scenario."
    ),
}


# ────────────────────────────────────────────────────────────
class NaradChatbot:
    def __init__(self):
        try:
            self.catalog = DebrisCatalog()
            self.ready = True
        except FileNotFoundError as e:
            print(f"[Narad] Warning: {e}")
            self.ready = False

    # ── Public API ──────────────────────────────────────────
    def ask(self, query: str) -> str:
        """
        Process a natural-language query about orbital objects.
        Returns a formatted markdown-style response string.
        """
        if not self.ready:
            return ("⚠ Catalog unavailable. Please ensure `data/debris_catalog.json` exists "
                    "and the system has been initialized.")

        q = query.strip()
        q_lower = q.lower()

        # ── Help ────────────────────────────────────────────
        if any(kw in q_lower for kw in ['help', 'what can you', 'what do you know', 'commands']):
            return HELP_TEXT

        # ── Type education queries ───────────────────────────
        for t, desc in TYPE_INFO.items():
            if f"what is a {t}" in q_lower or f"what are {t}" in q_lower or f"explain {t}" in q_lower:
                return f"**{t.upper()} — Object Classification**\n\n{desc}"

        # ── Stats query ─────────────────────────────────────
        if any(kw in q_lower for kw in ['how many', 'stats', 'statistics', 'total', 'count']):
            return self._stats_response(q_lower)

        # ── NORAD 5-digit ID ─────────────────────────────────
        norad_match = re.search(r'\b(\d{5})\b', q)
        if norad_match:
            obj_id = norad_match.group(1)
            obj = self.catalog.get_by_id(obj_id)
            if obj:
                return self.catalog.describe_object(obj)
            return f"⚠ No object found with NORAD ID **{obj_id}** in the catalog."

        # ── Short numeric ID (1–4 digits) ────────────────────
        short_id_match = re.search(r'\b(\d{1,4})\b', q)
        if short_id_match:
            obj_id = short_id_match.group(1).zfill(5)
            obj = self.catalog.get_by_id(obj_id)
            if obj:
                return self.catalog.describe_object(obj)

        # ── Alphanumeric ID (e.g. IE124, DEB-001, UK-SAT-7) ──
        alphanum_match = re.search(r'\b([A-Z]{1,4}[-]?\d{1,6})\b', q.upper())
        if alphanum_match:
            code = alphanum_match.group(1)
            # Search by name containing the code
            results = self.catalog.search_by_name(code)
            if results:
                obj = results[0]
                header = f"📡 Best match for **{code}**:\n\n"
                return header + self.catalog.describe_object(obj)
            return (
                f"⚠ No object matching '**{code}**' in catalog.\n"
                f"This might be a designation not in the current TLE dataset. "
                f"Try searching by NORAD ID or full name."
            )

        # ── Name search ──────────────────────────────────────
        if any(kw in q_lower for kw in ['what is', 'tell me about', 'info on', 'details about', 'find']):
            name_q = (q_lower
                      .replace('what is', '')
                      .replace('tell me about', '')
                      .replace('info on', '')
                      .replace('details about', '')
                      .replace('find', '')
                      .replace('debris', '')
                      .replace('satellite', '')
                      .replace('object', '')
                      .strip())
            if name_q:
                results = self.catalog.search_by_name(name_q, top_n=3)
                if results:
                    obj = results[0]
                    extra = ""
                    if len(results) > 1:
                        others = ", ".join(r['name'].strip() for r in results[1:])
                        extra = f"\n\n_Also matched: {others}_"
                    return self.catalog.describe_object(obj) + extra
                return (f"⚠ No object matching '**{name_q}**' found.\n"
                        f"Try a NORAD ID (e.g. '25544') or a fragment of the satellite name.")

        # ── Bare name / partial match fallback ───────────────
        if len(q.strip()) >= 3:
            results = self.catalog.search_by_name(q.strip(), top_n=3)
            if results:
                obj = results[0]
                header = f"📡 Best match for '**{q.strip()}**':\n\n"
                extra = ""
                if len(results) > 1:
                    others = ", ".join(r['name'].strip() for r in results[1:])
                    extra = f"\n\n_Also matched: {others}_"
                return header + self.catalog.describe_object(obj) + extra

        # ── Default fallback ─────────────────────────────────
        return (
            "I didn't understand that query. " + HELP_TEXT
        )

    # ── Internal Helpers ─────────────────────────────────────
    def _stats_response(self, q_lower: str) -> str:
        stats = self.catalog.get_stats()
        total = stats['total']
        types = stats['types']

        # Specific type count?
        for t in ['payload', 'debris', 'rocket', 'unknown']:
            if t in q_lower:
                count = types.get(t, 0)
                pct   = (count / total * 100) if total else 0
                return (
                    f"📊 **{t.upper()} Objects**\n\n"
                    f"  Count      : {count:,}\n"
                    f"  Percentage : {pct:.1f}% of all tracked objects\n"
                    f"  Total tracked : {total:,}"
                )

        # General stats
        breakdown = "\n".join(
            f"    {k.capitalize():<12} : {v:>6,}  ({v/total*100:.1f}%)"
            for k, v in sorted(types.items(), key=lambda x: -x[1])
        )
        return (
            f"📊 **OrbitalGuard Catalog Statistics**\n\n"
            f"  Total tracked objects : {total:,}\n\n"
            f"  Breakdown by type:\n{breakdown}\n\n"
            f"  ℹ  Data sourced from CelesTrak TLE dataset."
        )


# ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    bot = NaradChatbot()

    test_queries = [
        "What is debris 25544?",
        "Tell me about STARLINK-1234",
        "What type is IE124?",
        "How many objects are tracked?",
        "How many debris objects are there?",
        "What are rocket bodies?",
        "Help",
    ]

    print("\n" + "=" * 60)
    print("NARAD AI — Test Run")
    print("=" * 60)
    for q in test_queries:
        print(f"\n> {q}")
        print(bot.ask(q))
        print("-" * 40)
