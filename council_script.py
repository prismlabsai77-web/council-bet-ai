import asyncio
import os
import requests
from datetime import datetime
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# =========================
# 🔐 ENV VARIABLES
# =========================
SM_TOKEN = os.getenv("SPORTMONKS_API_KEY")
AF_KEY = os.getenv("SPORTS_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SM_TOKEN:
    print("❌ SPORTMONKS_API_KEY missing")
if not AF_KEY:
    print("❌ SPORTS_API_KEY missing")
if not SB_URL or not SB_KEY:
    print("❌ Supabase credentials missing")

supabase = create_client(SB_URL, SB_KEY)

# =========================
# 📡 FETCH FIXTURES
# =========================
def get_combined_fixtures():
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"📅 Fetching fixtures for UTC date: {date_str}")

    unified_data = {}

    # -------------------------
    # A. SPORTMONKS (PRIMARY)
    # -------------------------
    try:
        sm_url = "https://api.sportmonks.com/v3/football/fixtures"

        sm_res = requests.get(
            sm_url,
            params={
                "api_token": SM_TOKEN,
                "filters[date]": date_str,
                "include": "participants;league;venue"
            },
            timeout=15
        )

        print("Sportmonks Status:", sm_res.status_code)

        sm_json = sm_res.json()
        print("Sportmonks raw count:", len(sm_json.get("data", [])))

        for f in sm_json.get("data", []):
            home = next((p["name"] for p in f.get("participants", []) if p["meta"]["location"] == "home"), "Home")
            away = next((p["name"] for p in f.get("participants", []) if p["meta"]["location"] == "away"), "Away")

            name = f"{home} vs {away}"
            league = f.get("league", {}).get("name", "Unknown")

            unified_data[name] = {
                "game_id": name,
                "context": f"League: {league} | Venue: {f.get('venue', {}).get('name', 'TBD')} | Source: Sportmonks"
            }

    except Exception as e:
        print("⚠️ Sportmonks failed:", e)

    # -------------------------
    # B. API-FOOTBALL (FALLBACK)
    # -------------------------
    try:
        af_url = "https://v3.football.api-sports.io/fixtures"

        af_res = requests.get(
            af_url,
            headers={"x-apisports-key": AF_KEY},
            params={"date": date_str},
            timeout=15
        )

        print("API-Football Status:", af_res.status_code)

        af_json = af_res.json()
        print("API-Football raw count:", len(af_json.get("response", [])))

        for f in af_json.get("response", []):
            name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
            league = f["league"]["name"]
            referee = f["fixture"].get("referee", "TBD")

            if name not in unified_data:
                unified_data[name] = {
                    "game_id": name,
                    "context": f"League: {league} | Source: API-Football"
                }
            else:
                unified_data[name]["context"] += f" | Referee: {referee}"

    except Exception as e:
        print("⚠️ API-Football failed:", e)

    print(f"📡 Total unified matches: {len(unified_data)}")
    return unified_data


# =========================
# 🧠 MAIN COUNCIL RUNNER
# =========================
async def main():
    fixtures = get_combined_fixtures()

    if not fixtures:
        print("🛑 No matches found. Likely:")
        print("   • No fixtures scheduled today")
        print("   • Your plan doesn't include today's leagues")
        print("   • API timezone mismatch")
        return

    for name, data in fixtures.items():
        print(f"🗳️ Council debating: {name}")

        # Ensure match exists
        supabase.table("raw_fixtures").upsert({"game_id": name}).execute()

        try:
            results = await asyncio.gather(
                get_gemini_2_5_flash(data['context']),
                get_groq_qwen3(name),
                get_openrouter_glm45(name),
                get_hf_deepseek_r1(name),
                get_hf_kimi_k2(name),
                return_exceptions=True
            )

            payload = {
                "game_id": name,
                "gemini_2_5": str(results[0]),
                "qwen_3": str(results[1]),
                "glm_4_5": str(results[2]),
                "deepseek_r1": str(results[3]),
                "kimi_k2": str(results[4])
            }

            supabase.table("council_picks").insert(payload).execute()
            print("✅ Stored council vote")

        except Exception as e:
            print("⚠️ AI council failed for:", name, e)

    print("🎯 All matches processed.")


if __name__ == "__main__":
    asyncio.run(main())
