import asyncio
import os
import requests
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# --- INITIALIZATION ---
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
SM_TOKEN = os.getenv("SPORTMONKS_TOKEN")
AF_KEY = os.getenv("API_FOOTBALL_KEY") # Use Header: x-apisports-key

# --- 1. DATA AGGREGATION (SPORTMONKS + API-FOOTBALL) ---
def get_combined_fixtures():
    date_str = "2026-03-04"
    unified_data = {}

    # A. Fetch from Sportmonks (Primary)
    try:
        sm_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}"
        sm_res = requests.get(sm_url, params={"api_token": SM_TOKEN, "include": "participants;league;venue"}).json()
        for f in sm_res.get('data', []):
            name = f.get('name')
            league = f.get('league', {}).get('name', 'Unknown')
            unified_data[name] = {
                "game_id": name,
                "context": f"League: {league} | Venue: {f.get('venue', {}).get('name', 'TBD')} | Source: Sportmonks"
            }
    except Exception as e: print(f"⚠️ Sportmonks Fetch Failed: {e}")

    # B. Fetch from API-Football (Secondary/Supplement)
    try:
        af_url = "https://v3.football.api-sports.io/fixtures"
        headers = {"x-apisports-key": AF_KEY}
        af_res = requests.get(af_url, headers=headers, params={"date": date_str}).json()
        for f in af_res.get('response', []):
            name = f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}"
            ref = f['fixture'].get('referee', 'TBD')
            if name not in unified_data:
                unified_data[name] = {"game_id": name, "context": f"League: {f['league']['name']} | Source: API-Football"}
            else:
                unified_data[name]["context"] += f" | Referee: {ref}"
    except Exception as e: print(f"⚠️ API-Football Fetch Failed: {e}")

    return unified_data

# --- 2. THE GRAND COUNCIL (AI AGENTS) ---
async def get_gemini_2_5_flash(ctx):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Analyze {ctx}. Check latest injuries for March 4, 2026. Predict winner.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        return res.text
    except Exception as e: return f"Gemini Error: {str(e)[:50]}"

# [Keep other model functions: get_groq_qwen3, get_openrouter_glm45, get_hf_deepseek_r1, get_hf_kimi_k2 here]

# --- 3. MAIN LOOP ---
async def main():
    fixtures = get_combined_fixtures()
    print(f"📡 Found {len(fixtures)} matches for today (March 4, 2026).")

    for name, data in fixtures.items():
        print(f"🗳️ Council debating: {name}...")
        
        # Ensure match is in parent table
        supabase.table("raw_fixtures").upsert({"game_id": name}).execute()

        # Run 5-Model Council in Parallel
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

    print("✅ All votes recorded.")

if __name__ == "__main__":
    asyncio.run(main())
