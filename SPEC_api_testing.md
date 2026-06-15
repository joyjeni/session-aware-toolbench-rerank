# Spec: data.gov.in API Testing — Karnataka Smallholder Personas

## Goal
Conduct a thorough, auditable test of the **15 Karnataka government APIs** that the SessionRerank+ system retrieves over, acting as **three smallholder farmer personas**. Produce a **research-committee-ready evidence log** documenting every request, response, latency, and observed reliability issue.

## Entry point
Start at [https://www.data.gov.in/apis](https://www.data.gov.in/apis) — India's national open data portal. Find which of the 15 catalog APIs are accessible there, register if required (the user has Indian government login if needed — but try unauthenticated first).

## The 15 APIs to test
Exact catalog from `agri_extension/karnataka_apis.py`:

| # | API | URL/Endpoint |
|---|---|---|
| 1 | KSNDMC Weather Forecast | https://www.ksndmc.org |
| 2 | KSNDMC Drought Monitor | https://www.ksndmc.org/DroughtMonitoring.aspx |
| 3 | Bhoomi RTC (Record of Rights) | https://landrecords.karnataka.gov.in |
| 4 | Bhoochitra GIS Plot Viewer | https://landrecords.karnataka.gov.in/Service111/ |
| 5 | Raitha Mitra Crop Advisory | https://raitamitra.karnataka.gov.in |
| 6 | K-Kisan Toll-Free Helpline | tel:1551 (skip — not testable) |
| 7 | KSDA Pest & Disease Alert | https://raitamitra.karnataka.gov.in/PestAlerts |
| 8 | UAS-B / Plantix Offline Pest ID | https://plantix.net/en |
| 9 | Soil Health Card — Karnataka | https://soilhealth.dac.gov.in |
| 10 | Kannada NPK / FYM Calculator | https://uasbangalore.edu.in/npk-calculator |
| 11 | Agmarknet Karnataka Mandi Prices | https://agmarknet.gov.in (also via data.gov.in API) |
| 12 | ReMS (Unified Market Platform) | https://rems.karnataka.gov.in |
| 13 | PM-KISAN Beneficiary Status | https://pmkisan.gov.in |
| 14 | PMFBY Crop Insurance Enrolment | https://samrakshane.karnataka.gov.in |
| 15 | KMF Nandini Procurement Rate | https://kmfnandini.coop/farmer-rate |

For each API:
- Prefer the **data.gov.in JSON/XML endpoint** if available (search the data.gov.in catalog for the underlying provider, e.g., "Agmarknet" → catalog `dataset/dca0d20a-3812-4ff0-9b66-7c4a05cb4f6e`)
- Fall back to the **public web form / scraping endpoint** if no API exists
- Document **both** the official API URL and what we actually called

## The three personas

### Persona 1 — Ramesh, Tumakuru ragi farmer
- District: **Tumakuru**, Taluk: Sira
- Crop: **Ragi (finger millet)**, 2.5 acre rainfed plot, Kharif season
- Aadhaar-linked Bhoomi survey number (synthetic): `123/4A`
- Primary language: **Kannada**
- Phone: feature phone (no smartphone — affects API choice)
- Realistic queries:
  - "What's the ragi price at Tumakuru APMC today?"
  - "Has my PM-KISAN ₹2000 installment arrived?"
  - "Pest alert for ragi in Sira taluk this week"
  - "Soil health card status for survey 123/4A"

### Persona 2 — Lakshmi, Mandya sugarcane farmer
- District: **Mandya**, Taluk: Pandavapura
- Crop: **Sugarcane**, 5 acre irrigated (Cauvery canal), perennial
- Bhoomi survey: `45/2B`
- Primary language: **Kannada + some English**
- Phone: smartphone (Android, Plantix installed)
- Realistic queries:
  - "Sugarcane FRP rate for Mandya zone 2026"
  - "ReMS auction prices for sugarcane last week"
  - "Bhoochitra plot map for survey 45/2B"
  - "Yellow leaf disease photo — what is it?"

### Persona 3 — Suresh, Hassan coffee+coconut farmer
- District: **Hassan**, Taluk: Sakleshpur
- Crops: **Arabica coffee** (3 ac) + **coconut** (1.5 ac), hill agroclimatic zone
- Bhoomi survey: `78/1C`
- Primary language: **Kannada + English**
- Phone: smartphone, broadband at home
- Realistic queries:
  - "Coffee Board berry borer alert for Sakleshpur"
  - "Coconut price at Channarayapatna APMC"
  - "PMFBY enrolment status for coffee under WBCIS"
  - "Soil pH for coffee — what amendments needed?"

## What to log per API call (machine-readable)
A JSONL file `/home/user/workspace/api_test_log.jsonl` with one record per call:

```json
{
  "test_id": "T0001",
  "timestamp_ist": "2026-06-15T11:23:45+05:30",
  "persona": "ramesh_tumakuru_ragi",
  "api_name": "Agmarknet Karnataka Mandi Prices",
  "api_index": 11,
  "intended_query": "What's the ragi price at Tumakuru APMC today?",
  "endpoint_attempted": "https://api.data.gov.in/resource/9ef84268-...",
  "method": "GET",
  "headers_sent": {"User-Agent": "...", "Accept": "application/json"},
  "query_params": {"api-key": "REDACTED", "format": "json", "filters[state]": "Karnataka", "filters[district]": "Tumkur", "filters[commodity]": "Ragi (Finger Millet)"},
  "http_status": 200,
  "latency_ms": 847,
  "response_bytes": 12453,
  "response_sample": "{\"records\": [{\"state\": \"Karnataka\", \"district\": \"Tumkur\", \"market\": \"Tumkur\", \"commodity\": \"Ragi(Finger Millet)\", \"modal_price\": \"3650\", \"price_date\": \"2026-06-14\"}, ...]}",
  "parsed_answer": "₹3650/quintal at Tumkur APMC, dated 2026-06-14",
  "data_freshness": "1 day old",
  "verdict": "WORKING",
  "issues_observed": ["price_date 1 day stale", "no Sira sub-market data"],
  "screenshot_path": null
}
```

Verdict enum: `WORKING`, `DEGRADED` (response received but incomplete/stale), `BROKEN` (5xx, timeout, malformed), `BLOCKED` (requires auth we don't have), `NOT_FOUND` (endpoint dead), `SKIPPED` (telephone-only, e.g., K-Kisan helpline).

## What to produce (deliverables)

1. **Machine-readable log**: `/home/user/workspace/api_test_log.jsonl` — one row per API call (3 personas × 4 queries each = ~36 calls plus exploratory data.gov.in catalog searches)

2. **Committee-ready PDF report**: `/home/user/workspace/karnataka_api_audit.pdf`, ~10-15 pages, with:
   - Cover page (audit date, scope, three personas)
   - Executive summary (X/15 APIs working, Y degraded, Z broken)
   - Per-API section (1 page each, with: endpoint, sample request, sample response, latency, freshness, verdict, recommendation)
   - Per-persona section showing the realistic farmer journey across multiple APIs
   - Reproducibility appendix — exact curl commands for every successful call, so the research committee can re-run

3. **CSV summary**: `/home/user/workspace/api_audit_summary.csv` — flat table with columns `api_name, persona, query, http_status, latency_ms, verdict, issue_count`

4. **Screenshots**: For every web-form-based API (no JSON endpoint), capture a screenshot of the working interaction. Save to `/home/user/workspace/api_screenshots/` and reference paths in the JSONL log.

## How to test

### Browser-based (for sites without JSON APIs)
Use `browser_task` with detailed step-by-step navigation. Take screenshots. The form-based sites are likely:
- Bhoomi RTC (form-based, requires survey number)
- Bhoochitra (GIS viewer)
- Soil Health Card (form + Aadhaar)
- PM-KISAN (beneficiary form)
- PMFBY (Samrakshane portal)

### API-based (for data.gov.in catalog entries)
Use `bash` + `curl`. Many require `api-key` query param — try the public demo key `579b464db66ec23bdd00000139b0d95a6ee4441f5cd8e7ad6f59b8a8` (this is data.gov.in's public testing key, documented and rate-limited).

### Wide-browse for catalog discovery
Use `wide_browse` to systematically probe data.gov.in for each catalog name (e.g., "Agmarknet", "Bhoomi", "PMFBY"). Save results to CSV first.

## Constraints
- **Do NOT send any actual SMS, place real phone calls, submit real PMFBY enrolments, or submit any real form that creates state on a government system.** Read-only / query-only.
- The Bhoomi survey numbers above (123/4A, 45/2B, 78/1C) are **synthetic** — do not enter real owner names from public records and store them in our log; redact PII even from sample responses (replace names with `[NAME_REDACTED]`).
- For any API that requires authentication (Aadhaar OTP, DigiLocker, etc.), **stop at the auth wall** — log it as `BLOCKED`, do not attempt to bypass.
- Total scope: ~36-50 API calls. Should take a single session ~1-2 hours.

## Final hand-off
At the end, send a single notification with a 5-line summary:
```
Karnataka API Audit complete.
- N/15 APIs WORKING, M DEGRADED, P BROKEN, Q BLOCKED.
- Headline finding: <one sentence>.
- Full log: /workspace/api_test_log.jsonl
- Committee PDF: /workspace/karnataka_api_audit.pdf
- CSV summary: /workspace/api_audit_summary.csv
```
And `share_file` the PDF + CSV.
