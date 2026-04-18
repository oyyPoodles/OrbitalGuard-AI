"""
OrbitalGuard AI — Debris Catalog
=================================
Loads and indexes the debris_catalog.json for fast retrieval by:
  - NORAD ID (exact integer or string)
  - Alphanumeric object ID (e.g. "IE124")
  - Name keyword (partial / case-insensitive)
  - Object type (payload / debris / rocket)

Catalog entry schema:
  {
    "id":         "25544",    # 5-digit NORAD string
    "name":       "ISS (ZARYA)",
    "type":       "payload",
    "epoch_yr":   26,
    "epoch_days": 57.31
  }
"""

import os
import json
import re
from difflib import get_close_matches
from typing import Optional, List, Dict, Any

# ─── Path ────────────────────────────────────────────────
KNOWLEDGE_DIR  = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT      = os.path.dirname(KNOWLEDGE_DIR)
CATALOG_PATH   = os.path.join(AIML_ROOT, 'data', 'debris_catalog.json')


# ─── Object Type Labels ──────────────────────────────────
TYPE_DESCRIPTIONS = {
    'payload':  'Active or inactive satellite payload',
    'debris':   'Space debris / fragmentation object',
    'rocket':   'Rocket body / upper stage',
    'unknown':  'Unclassified orbital object',
}

RISK_NOTES = {
    'payload':  'Actively tracked; conjunction assessments applied.',
    'debris':   'High collision risk contributor; monitored continuously.',
    'rocket':   'Derelict rocket body; significant cross-section.',
    'unknown':  'Classification pending; treat as potential hazard.',
}


# ────────────────────────────────────────────────────────
class DebrisCatalog:
    """
    In-memory catalog of all tracked orbital objects.
    Provides O(1) lookup by ID and fuzzy search by name.
    """

    def __init__(self, catalog_path: str = CATALOG_PATH):
        self._by_id:   Dict[str, Dict]  = {}   # NORAD ID (str) → entry
        self._by_name: List[str]         = []   # lowercased names for fuzzy matching
        self._entries: List[Dict]        = []
        self._load(catalog_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[DebrisCatalog] Catalog not found: {path}\n"
                f"Expected: data/debris_catalog.json"
            )

        with open(path, 'r', encoding='utf-8') as f:
            raw: List[Dict] = json.load(f)

        for entry in raw:
            # Normalize ID to string
            obj_id = str(entry.get('id', '')).strip().zfill(5)
            entry['id'] = obj_id
            entry['name'] = str(entry.get('name', 'UNKNOWN')).strip()
            entry.setdefault('type', 'unknown')
            entry.setdefault('epoch_yr', 0)
            entry.setdefault('epoch_days', 0.0)

            self._by_id[obj_id] = entry
            self._entries.append(entry)
            self._by_name.append(entry['name'].lower())

        print(f"[DebrisCatalog] Loaded {len(self._entries)} objects.")

    # ── Retrieval ─────────────────────────────────────────
    def get_by_id(self, obj_id: str) -> Optional[Dict]:
        """
        Retrieve object by NORAD ID.
        Accepts: "25544", 25544, "ISS", partial names.
        Supports zero-padded IDs automatically.
        """
        # Try exact 5-digit lookup
        padded = str(obj_id).strip().zfill(5)
        if padded in self._by_id:
            return self._by_id[padded]

        # Try unpadded
        raw = str(obj_id).strip()
        if raw in self._by_id:
            return self._by_id[raw]

        return None

    def search_by_name(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Search by name keyword (case-insensitive, partial + fuzzy).
        Returns up to top_n matching objects.
        """
        q = query.strip().lower()
        results = []

        # 1. Exact substring match (fast)
        exact = [e for e in self._entries if q in e['name'].lower()]
        results.extend(exact)

        # 2. Fuzzy match (difflib)
        if len(results) < top_n:
            fuzzy_names = get_close_matches(q, self._by_name, n=top_n, cutoff=0.5)
            for fn in fuzzy_names:
                idx = self._by_name.index(fn)
                candidate = self._entries[idx]
                if candidate not in results:
                    results.append(candidate)

        return results[:top_n]

    def get_by_type(self, obj_type: str) -> List[Dict]:
        """Retrieve all objects of a given type."""
        return [e for e in self._entries if e.get('type', '').lower() == obj_type.lower()]

    def get_stats(self) -> Dict[str, Any]:
        """Return catalog-wide statistics."""
        types: Dict[str, int] = {}
        for e in self._entries:
            t = e.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1

        return {
            'total': len(self._entries),
            'types': types,
        }

    def describe_object(self, entry: Dict) -> str:
        """
        Generate a rich natural-language description of a single catalog object.
        Used by NaradChatbot for formatted responses.
        """
        obj_type  = entry.get('type', 'unknown')
        epoch_yr  = entry.get('epoch_yr', '??')
        epoch_day = entry.get('epoch_days', '??')

        year_str = f"20{epoch_yr:02d}" if isinstance(epoch_yr, int) else str(epoch_yr)
        day_str  = f"{float(epoch_day):.2f}" if epoch_day != '??' else '??'

        lines = [
            f"🛰  **{entry['name']}**  (NORAD ID: {entry['id']})",
            f"",
            f"  Classification : {obj_type.upper()} — {TYPE_DESCRIPTIONS.get(obj_type, '')}",
            f"  Epoch Year     : {year_str}",
            f"  Epoch Day      : {day_str}",
            f"",
            f"  ⚠  Risk Note   : {RISK_NOTES.get(obj_type, 'Monitor continuously.')}",
        ]
        return "\n".join(lines)

    def all_names(self) -> List[str]:
        """Return list of all object names (for frontend satellite list)."""
        return [e['name'] for e in self._entries]

    def all_entries(self) -> List[Dict]:
        """Return full entry list (for API /api/satellites endpoint)."""
        return self._entries


# ────────────────────────────────────────────────────────
if __name__ == '__main__':
    catalog = DebrisCatalog()
    stats = catalog.get_stats()
    print(f"\nTotal tracked objects : {stats['total']}")
    print(f"Type breakdown        : {stats['types']}")

    # Test by ID
    iss = catalog.get_by_id('25544')
    if iss:
        print(f"\nLookup 25544:\n{catalog.describe_object(iss)}")

    # Test by name
    results = catalog.search_by_name('starlink')
    print(f"\nSearch 'starlink': {len(results)} results")
    if results:
        print(f"  First: {results[0]['name']} (ID: {results[0]['id']})")
