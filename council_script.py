import asyncio
import os
import requests
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
            contents=f"Betting analysis (1, X, 2): {context}"
        )
        return response.text
    except Exception as e: return f"Error: {e}"

async def get_groq_qwen3(name):
    try:
        client = AsyncGroq(api_key=GROQ_KEY)
        chat = await client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[{"role": "user", "content": f"Winner for {name}?"}]
        )
        return chat.choices[0].message.content
    except Exception as e: return f"Error: {e}"

async def get_openrouter_glm45(name):
    try:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        res = await client.chat.completions.create(
            model="zhipuai/glm-4-9b-chat",
            messages=[{"role": "user", "content": f"Analysis for {name}"}]
        )
        return res.choices[0].message.content
    except Exception as e: return f"Error: {e}"

async def get_hf_deepseek_r1(name):
    try:
        url = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        res = requests.post(url, headers=headers, json={"inputs": name}, timeout=15)
        return res.json()[0]['generated_text'] if res.status_code == 200 else "Busy"
    except Exception: return "Error"

async def get_hf_kimi_k2(name):
    try:
        url = "https://api-inference.huggingface.co/models/moonshotai/Kimi-K2-Instruct"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        res = requests.post(url, headers=headers, json={"inputs": name}, timeout=15)
        return res.json()[0]['generated_text'] if res.status_code == 200 else "Busy"
    except Exception: return "Error"

# =========================
# 📡 FETCH FIXTURES
# =========================
def get_combined_fixtures():
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    unified_data = {}

    # Sportmonks
    try:
        sm_res = requests.get(f"https://api.sportmonks.com/v3/football/fixtures/date/{date_str}", 
                              params={"api_token": SM_TOKEN, "include": "participants;league"}, timeout=15)
        if sm_res.status_code == 200:
            for f in sm_res.json().get("data", []):
                h = next((p['name'] for p in f['participants'] if p['meta']['location'] == 'home'), "Home")
                a = next((p['name'] for p in f['participants'] if p['meta']['location'] == 'away'), "Away")
                name = f"{h} vs {a}"
                unified_data[name] = {"home": h, "away": a, "kickoff": f.get("starting_at"), 
                                      "league": f.get("league", {}).get("name"), "context": name}
    except Exception as e: print(f"SM Error: {e}")

    # API-Football
    try:
        af_res = requests.get("https://v3.football.api-sports.io/fixtures", 
                              headers={"x-apisports-key": AF_KEY}, params={"date": date_str}, timeout=15)
        if af_res.status_code == 200:
            for f in af_res.json().get("response", []):
                h, a = f['teams']['home']['name'], f['teams']['away']['name']
                name = f"{h} vs {a}"
                if name not in unified_data:
                    unified_data[name] = {"home": h, "away": a, "kickoff": f['fixture'].get('date'), 
                                          "league": f['league']['name'], "context": name}
    except Exception as e: print(f"AF Error: {e}")

    return unified_data

# =========================
# 🧠 MAIN RUNNER
# =========================
async def main():
    fixtures = get_combined_fixtures()
    if not fixtures: return

    for name, data in list(fixtures.items())[:10]:
        print(f"🗳️ Council debating: {name}")
        try:
            # 1. Upsert Fixture
            supabase.table("raw_fixtures").upsert({
                "game_id": name,
                "sport_type": "football",
                "home_team": data['home'],
                "away_team": data['away'],
                "league_name": data['league'],
                "kickoff_time": data['kickoff']
            }).execute()

            # 2. Parallel AI Calls
            results = await asyncio.gather(
                get_gemini_2_5_flash(data['context']),
                get_groq_qwen3(name),
                get_openrouter_glm45(name),
                get_hf_deepseek_r1(name),
                get_hf_kimi_k2(name),
                return_exceptions=True
            )

            # 3. Insert Picks
            supabase.table("council_picks").insert({
                "game_id": name,
                "sport_type": "football", # This was causing your latest error
                "gemini_2_5": str(results[0]),
                "qwen_3": str(results[1]),
                "glm_4_5": str(results[2]),
                "deepseek_r1": str(results[3]),
                "kimi_k2": str(results[4])
            }).execute()

            print(f"✅ Stored: {name}")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
