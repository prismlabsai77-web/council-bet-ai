import asyncio
import os
import requests
from datetime import datetime
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# --- CLIENTS ---
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
SM_TOKEN = os.getenv("SPORTMONKS_TOKEN")
AF_KEY = os.getenv("API_FOOTBALL_KEY") # Headers: x-apisports-key

# --- DATA AGGREGATION ---
def get_combined_today_data():
    date_str = "2026-03-04"
    unified_fixtures = {}

    # 1. Fetch from Sportmonks (Primary)
    try:
        sm_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}"
        sm_res = requests.get(sm_url, params={"api_token": SM_TOKEN, "include": "participants;venue"}).json()
        for f in sm_res.get('data', []):
            name = f.get('name')
            venue = f.get('venue', {}).get('name', 'Unknown')
            unified_fixtures[name] = {
                "game_id": name,
                "context": f"Source: Sportmonks | Venue: {venue} | Status: {f.get('result_info', 'Scheduled')}"
            }
    except Exception as e: print(f"⚠️ Sportmonks Error: {e}")

    # 2. Fetch from API-Football (Fallback/Supplement)
    try:
        af_url = "https://v3.football.api-sports.io/fixtures"
        headers = {"x-apisports-key": AF_KEY}
        af_res = requests.get(af_url, headers=headers, params={"date": date_str}).json()
        for f in af_res.get('response', []):
            name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
            referee = f['fixture'].get('referee', 'TBD')
            league = f['league'].get('name', 'General')
            
            if name not in unified_fixtures:
                # Add missing game
                unified_fixtures[name] = {
                    "game_id": name,
                    "context": f"Source: API-Football | League: {league} | Referee: {referee}"
                }
            else:
                # Supplement existing game data
                unified_fixtures[name]["context"] += f" | Referee: {referee} | League: {league}"
    except Exception as e: print(f"⚠️ API-Football Error: {e}")

    return unified_fixtures

# --- AI AGENTS (Parallel) ---
# [Ensure your get_gemini_2_5_flash, get_groq_qwen3, etc., functions are defined here as before]

async def main():
    fixtures = get_combined_today_data()
    print(f"📡 Found {len(fixtures)} unified matches for today.")

    for name, data in fixtures.items():
        print(f"🗳️ Council voting for: {name}...")
        
        # Ensure match exists in parent table
        supabase.table("raw_fixtures").upsert({"game_id": name}).execute()

        # Run Council in Parallel
        # (Using the combined 'context' string to give AI more info)
        results = await asyncio.gather(
            get_gemini_2_5_flash(f"{name}. {data['context']}"),
            get_groq_qwen3(name),
            get_openrouter_glm45(name),
            get_hf_deepseek_r1(name),
            get_hf_kimi_k2(name),
            return_exceptions=True
        )

        # Prepare Payload
        payload = {
            "game_id": name,
            "gemini_2_5": str(results[0]),
            "qwen_3": str(results[1]),
            "glm_4_5": str(results[2]),
            "deepseek_r1": str(results[3]),
            "kimi_k2": str(results[4])
        }

        try:
            supabase.table("council_picks").insert(payload).execute()
            print(f"✅ Success for {name}")
        except Exception as e:
            print(f"❌ Record Error for {name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
