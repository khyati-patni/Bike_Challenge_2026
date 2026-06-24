"""
Stage 0 · stage0_fetch_abstracts.py
─────────────────────────────────────
Fetches title + abstract for every unique DOI across all 5 tasks
using the free CrossRef API.  Results cached in data/doi_texts.json.
Re-running is safe — already-fetched DOIs are skipped.

Output: data/doi_texts.json  {doi: "title. abstract"}

Run this BEFORE stage1_encode_embeddings.py.
"""
import json, time, re, pathlib, sys
import requests
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import TASKS, DATA_DIR

OUT = DATA_DIR / "doi_texts.json"

# CrossRef polite pool settings (adjust as needed)
CROSSREF_EMAIL = "your@email.com"   # set to your email for polite pool access
FETCH_TIMEOUT  = 15                 # seconds per request
FETCH_RETRIES  = 3                  # max retries on failure


def fetch_doi(doi: str) -> str:
    url     = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"BiKEv2Pipeline/2.0 (mailto:{CROSSREF_EMAIL})"}
    for attempt in range(FETCH_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            msg    = r.json().get("message", {})
            title  = " ".join(msg.get("title", [""]))
            abstr  = msg.get("abstract", "")
            abstr  = re.sub(r"<[^>]+>", " ", abstr)   # strip JATS XML tags
            return f"{title}. {abstr}".strip()
        except Exception as e:
            if attempt < FETCH_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [!] {doi}: {e}")
                return ""
    return ""


def all_dois() -> list:
    dois = set()
    for cfg in TASKS.values():
        for split in ("train", "test"):
            dois.update(pd.read_csv(cfg[split])["node"].unique())
    return sorted(dois)


def main():
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    print(f"[stage 0] cache has {len(cache)} DOIs")

    missing = [d for d in all_dois() if d not in cache]
    print(f"[stage 0] {len(missing)} DOIs to fetch from CrossRef\n")

    for i, doi in enumerate(missing, 1):
        text = fetch_doi(doi)
        cache[doi] = text
        print(f"  {i:3}/{len(missing)}  {'ok' if text else '--'}  {doi[:50]}")
        if i % 20 == 0:
            OUT.write_text(json.dumps(cache, indent=2))
        time.sleep(0.13)   # ~7 req/sec, well within CrossRef limits

    OUT.write_text(json.dumps(cache, indent=2))
    filled = sum(1 for v in cache.values() if v)
    print(f"\n[stage 0 done] {filled}/{len(cache)} DOIs have text → {OUT}")


if __name__ == "__main__":
    main()
