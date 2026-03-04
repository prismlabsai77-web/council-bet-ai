import asyncio
import os
import requests
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# --- INITIALIZATION (ALIGNED WITH YOUR SECRET NAMES) ---
# Pulling the EXACT names from your GitHub Secrets repository
SM_TOKEN = os.getenv("SPORTMONKS_API_KEY") # Matches your 'SPORTMONKS_API_KEY'
AF_KEY = os.getenv("SPORTS_API_KEY")       # Matches your 'SPORTS_API_KEY'
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize Supabase
supabase = create_client(SB_URL, SB_KEY)

# --- 1. DATA AGGREGATION (WITH FALLBACK LOGIC) ---
def get_combined_fixtures():
    # Today is March 4, 2026
    date_str = "2026-03-04"
    unified_data = {}

    # A. Fetch from Sportmonks (Primary)
    try:
        sm_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}"
        # Setting a timeout and checking status
        sm_res = requests.get(sm_url, params={"api_token": SM_TOKEN, "include": "participants;league;venue"}, timeout=10)
        sm_json = sm_res.json()
        
        for f in sm_json.get('data', []):
            name = f.get('name')
            league = f.get('league', {}).get('name', 'Unknown')
            unified_data[name] = {
                "game_id": name,
                "context": f"League: {league} | Venue: {f.get('venue', {}).get('name', 'TBD')} | Source: Sportmonks"
            }
    except Exception as e: 
        print(f"⚠️ Sportmonks failed or key is missing: {e}")

    # B. Fetch from Sports API (API-Football) (Secondary/Fallback)
    try:
        af_url = "https://v3.football.api-sports.io/fixtures"
        headers = {"x-apisports-key": AF_KEY} # Uses your SPORTS_API_KEY
        af_res = requests.get(af_url, headers=headers, params={"date": date_str}, timeout=10)
        af_json = af_res.json()
        
        for f in af_json.get('response', []):
            name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
            ref = f['fixture'].get('referee', 'TBD')
            
            # If Sportmonks missed it, add it. If it exists, add the referee data.
            if name not in unified_data:
                unified_data[name] = {
                    "game_id": name, 
                    "context": f"League: {f['league']['name']} | Source: Sports API (Fallback)"
                }
            else:
                unified_data[name]["context"] += f" | Referee: {ref}"
    except Exception as e: 
        print(f"⚠️ Sports API failed or key is missing: {e}")

    return unified_data

# --- [Keep your AI Agent functions (Gemini, Groq, etc.) here] ---

async def main():
    fixtures = get_combined_fixtures()
    print(f"📡 Successfully pulled {len(fixtures)} matches using unified keys.")

    if not fixtures:
        print("🛑 No matches found. Check your API plan or verify date: 2026-03-04.")
        return

    for name, data in fixtures.items():
        print(f"🗳️ Council debating: {name}...")
        
        # Ensure match is in parent table
        supabase.table("raw_fixtures").upsert({"game_id": name}).execute()

        # Run 5-Model Council in Parallel (Pass the unified context to the AI)
        results = await asyncio.gather(
            get_gemini_2_5_flash(data['context']),
            get_groq_qwen3(name),
            get_openrouter_glm45(name),
            get_hf_deepseek_r1(name),
            get_hf_kimi_k2(name),
            return_exceptions=True
        )

        # Store in Supabase
        payload = {
            "game_id": name,
            "gemini_2_5": str(results[0]),
            "qwen_3": str(results[1]),
            "glm_4_5": str(results[2]),
            "deepseek_r1": str(results[3]),
            "kimi_k2": str(results[4])
        }
        supabase.table("council_picks").insert(payload).execute()

    print("✅ All votes recorded in Supabase.")

if __name__ == "__main__":
    asyncio.run(main())
