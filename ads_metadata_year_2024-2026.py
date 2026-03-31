import os
import gzip
import json
import requests
from datetime import datetime
from time import sleep

# ─── CONFIG ────────────────────────────────────────────────────────────────
API_KEY = os.environ["ADS_API_KEY"]
API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
FIELDS = ",".join([
    "abstract", "ack", "aff", "alternate_bibcode", "alternate_title",
    "arxiv_class", "author", "bibcode", "bibgroup", "bibstem",
    "body", "citation", "citation_count", "copyright", "database",
    "data", "doi", "doctype", "editor", "entry_date", "first_author",
    "grant", "id", "identifier", "indexstamp", "issue", "keyword",
    "lang", "orcid_pub", "orcid_user", "page", "property",
    "pub", "pub_raw", "pubdate", "read_count", "reference",
    "reference_count", "series", "title", "volume", "year"
])
OUTPUT_DIR = "ads_metadata_by_year_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ROWS = 100
TIMEOUT = 60
THROTTLE = 1  # seconds between batches

# ─── LOGGING ───────────────────────────────────────────────────────────────
def log(msg):
    print(f"{datetime.utcnow().isoformat()} | {msg}")

# ─── FILE CHECK ────────────────────────────────────────────────────────────
def count_lines(path):
    if not os.path.exists(path):
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)

# ─── FETCH BATCH ───────────────────────────────────────────────────────────
def fetch_batch(year, start, rows):
    query = {
        "q": f"*:*  AND year:{year}",
        "start": start,
        "rows": rows,
        "fl": FIELDS
    }
    attempt = 0
    while True:
        try:
            resp = requests.get(API_URL, headers=HEADERS, params=query, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("response", {}).get("docs", [])
            else:
                log(f"⚠️ HTTP {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            log(f"🌐 Attempt {attempt+1} failed: {e}")
        attempt += 1
        sleep(min(60, 2 ** min(attempt, 6)))  # retry with backoff (max 60s)

# ─── MAIN YEAR RETRIEVAL ───────────────────────────────────────────────────
def retrieve_year(year):
    output_path = os.path.join(OUTPUT_DIR, f"ads_metadata_{year}_full.jsonl.gz")
    start = count_lines(output_path)
    log(f"📆 {year}: Starting from {start} records...")

    with gzip.open(output_path, "at", encoding="utf-8") as f:
        while True:
            log(f"🔎 Fetching {ROWS} records from {start} for {year}...")
            records = fetch_batch(year, start, ROWS)
            if not records:
                log(f"✅ Finished {year}")
                break
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            start += ROWS
            sleep(THROTTLE)

# ─── RUN FULL RANGE (1940–2026) ────────────────────────────────────────────
def run_all_years(start_year=2024, end_year=2026):
    for year in range(start_year, end_year + 1):
        while True:
            try:
                retrieve_year(year)
                break
            except Exception as e:
                log(f"💥 Error in year {year}: {e} — retrying in 60s")
                sleep(60)

# ─── LAUNCH ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_years(2024, 2026)

