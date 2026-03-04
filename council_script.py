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
# 🔐 ENV VARIABLES (Aligned with your GitHub Secrets)
# =========================
SM_TOKEN = os.getenv("SPORTMONKS_API_KEY")
AF_KEY = os.getenv("SPORTS_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SB_URL, SB_KEY)

# =========================
# 📡 FETCH FIXTURES
# =========================
def get_combined_fixtures():
    # Use UTC to align with API servers
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"📅 Fetching fixtures for UTC date: {date_str}")

    unified_data = {}

    # -------------------------
    # A. SPORTMONKS (v3 Path Fix)
    # -------------------------
    try:
        # v3 specifically likes the date in the URL path for better stability
        sm_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}"
        sm_res = requests.get(
            sm_url,
            params={
                "api_token": SM_TOKEN,
                "include": "participants;league;venue"
            },
            timeout=15
        )
        print(f"Sportmonks Status: {sm_res.status_code}")
        
        if sm_res.status_code == 200:
            sm_json = sm_res.json()
            fixtures = sm_json.get("data", [])
            print(f"Sportmonks raw count: {len(fixtures)}")

            for f in fixtures:
                # v3 participants parsing
                home = "Home"
                away = "Away"
                for p in f.get("participants", []):
                    if p.get("meta", {}).get("location") == "home":
                        home = p.get("name")
                    elif p.get("meta", {}).get("location") == "away":
                        away = p.get("name")

                name = f"{home} vs {away}"
                league = f.get("league", {}).get("name", "Unknown")
                unified_data[name] = {
                    "game_id": name,
                    "context": f"League: {league} | Venue: {f.get('venue', {}).get('name', 'TBD')} | Source: Sportmonks"
                }
    except Exception as e:
        print(f"⚠️ Sportmonks failed: {e}")

    # -------------------------
    # B. API-FOOTBALL (Fallback)
    # -------------------------
    try:
        af_url = "https://v3.football.api-sports.io/fixtures"
        af_res = requests.get(
            af_url,
            headers={"x-apisports-key": AF_KEY},
            params={"date": date_str},
            timeout=15
        )
        print(f"API-Football Status: {af_res.status_code}")
        
        if af_res.status_code == 200:
            af_json = af_res.json()
            fixtures = af_json.get("response", [])
            print(f"API-Football raw count: {len(fixtures)}")

            for f in fixtures:
                name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
                if name not in unified_data:
                    unified_data[name] = {
                        "game_id": name,
                        "context": f"League: {f['league']['name']} | Source: API-Football"
                    }
                else:
                    unified_data[name]["context"] += f" | Referee: {f['fixture'].get('referee', 'TBD')}"
    except Exception as e:
        print(f"⚠️ API-Football failed: {e}")

    return unified_data

# =========================
# 🧠 MAIN COUNCIL RUNNER
# =========================
async def main():
    fixtures = get_combined_fixtures()

    if not fixtures:
        print("🛑 No matches found today.")
        return

    # To avoid processing 300+ matches, let's limit to top 10 or filter by league
    # Example: fixtures = {k: v for k, v in list(fixtures.items())[:10]}

    for name, data in fixtures.items():
        print(f"🗳️ Council debating: {name}")

        # --- FIX: Satisfy the "sport_type" NOT NULL constraint ---
        try:
            supabase.table("raw_fixtures").upsert({
                "game_id": name,
                "sport_type": "football" # Added mandatory column
            }).execute()

            # AI parallel calls (Assuming functions are defined below)
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
                "sport_type": "football", # Added mandatory column here too
                "gemini_2_5": str(results[0]),
                "qwen_3": str(results[1]),
                "glm_4_5": str(results[2]),
                "deepseek_r1": str(results[3]),
                "kimi_k2": str(results[4])
            }

            supabase.table("council_picks").insert(payload).execute()
            print(f"✅ Stored picks for {name}")

        except Exception as e:
            print(f"⚠️ Failed to process {name}: {e}")

    print("🎯 Process complete.")

if __name__ == "__main__":
    asyncio.run(main())
