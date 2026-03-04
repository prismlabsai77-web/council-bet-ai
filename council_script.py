import asyncio
import os
import requests
import json
from datetime import datetime
from google import genai
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

# Model Keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

supabase = create_client(SB_URL, SB_KEY)

# =========================
# 🧠 AI COUNCIL MODELS
# =========================

async def get_gemini_2_5_flash(context):
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=f"Analyze this football match and provide a betting prediction (1, X, or 2) with a brief reason: {context}"
        )
        return response.text
    except Exception as e: return f"Gemini Error: {e}"

async def get_groq_qwen3(match_name):
    try:
        client = AsyncGroq(api_key=GROQ_KEY)
        chat = await client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": f"Predict the outcome for {match_name}. Be concise."}]
        )
        return chat.choices[0].message.content
    except Exception as e: return f"Groq Error: {e}"

async def get_openrouter_glm45(match_name):
    try:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY,
        )
        response = await client.chat.completions.create(
            model="zhipuai/glm-4-9b-chat",
            messages=[{"role": "user", "content": f"Provide a betting analysis for {match_name}."}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenRouter Error: {e}"

async def get_hf_deepseek_r1(match_name):
    try:
        API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": f"Think step by step: Who wins {match_name}?"}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        return response.json()[0]['generated_text'] if response.status_code == 200 else "DeepSeek Busy"
    except Exception as e: return f"DeepSeek Error: {e}"

async def get_hf_kimi_k2(match_name):
    try:
        API_URL = "https://api-inference.huggingface.co/models/moonshotai/Kimi-K2-Instruct"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": f"Analyze {match_name} and pick a winner."}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        return response.json()[0]['generated_text'] if response.status_code == 200 else "Kimi Busy"
    except Exception as e: return f"Kimi Error: {e}"

# =========================
# 📡 FETCH FIXTURES
# =========================
def get_combined_fixtures():
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"📅 Fetching fixtures for: {date_str}")
    unified_data = {}

    # Sportmonks v3
    try:
        sm_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}"
        sm_res = requests.get(sm_url, params={"api_token": SM_TOKEN, "include": "participants;league"}, timeout=15)
        if sm_res.status_code == 200:
            for f in sm_res.json().get("data", []):
                home, away = "Home", "Away"
                for p in f.get("participants", []):
                    if p.get("meta", {}).get("location") == "home": home = p.get("name")
                    elif p.get("meta", {}).get("location") == "away": away = p.get("name")
                
                name = f"{home} vs {away}"
                unified_data[name] = {
                    "home": home, 
                    "away": away, 
                    "kickoff": f.get("starting_at"), # Sportmonks V3 field
                    "league": f.get("league", {}).get("name", "Unknown"),
                    "context": f"League: {f.get('league', {}).get('name')} | Kickoff: {f.get('starting_at')}"
                }
    except Exception as e: print(f"⚠️ Sportmonks failed: {e}")

    # API-Football
    try:
        af_res = requests.get("https://v3.football.api-sports.io/fixtures", 
                              headers={"x-apisports-key": AF_KEY}, params={"date": date_str}, timeout=15)
        if af_res.status_code == 200:
            for f in af_res.json().get("response", []):
                h, a = f['teams']['home']['name'], f['teams']['away']['name']
                name = f"{h} vs {a}"
                if name not in unified_data:
                    unified_data[name] = {
                        "home": h, 
                        "away": a, 
                        "kickoff": f['fixture'].get('date'), # API-Football field
                        "league": f['league']['name'],
                        "context": f"League: {f['league']['name']} | Kickoff: {f['fixture'].get('date')}"
                    }
    except Exception as e: print(f"⚠️ API-Football failed: {e}")

    return unified_data

# =========================
# 🧠 MAIN RUNNER
# =========================
async def main():
    fixtures = get_combined_fixtures()
    if not fixtures:
        print("🛑 No matches found.")
        return

    # To test, we process the first 10 matches
    for name, data in list(fixtures.items())[:10]:
        print(f"🗳️ Council debating: {name}")

        try:
            # 1. Update/Insert raw fixture including kickoff_time
            supabase.table("raw_fixtures").upsert({
                "game_id": name,
                "sport_type": "football",
                "home_team": data['home'],
                "away_team": data['away'],
                "league_name": data['league'],
                "kickoff_time": data['kickoff'] # Fixed the null error
            }).execute()

            # 2. Run all 5 AI models in parallel
            results = await asyncio.gather(
                get_gemini_2_5_flash(data['context']),
                get_groq_qwen3(name),
                get_openrouter_glm45(name),
                get_hf_deepseek_r1(name),
                get_hf_kimi_k2(name),
                return_exceptions=True
            )

            # 3. Store the Council results in picks table
            payload = {
                "game_id": name,
                "sport_type": "football",
                "gemini_2_5": str(results[0]),
                "qwen_3": str(results[1]),
                "glm_4_5": str(results[2]),
                "deepseek_r1": str(results[3]),
                "kimi_k2": str(results[4])
            }

            supabase.table("council_picks").insert(payload).execute()
            print(f"✅ Full council picks stored for {name}")

        except Exception as e:
            print(f"⚠️ Error processing {name}: {e}")

    print("🎯 Process complete.")

if __name__ == "__main__":
    asyncio.run(main())
