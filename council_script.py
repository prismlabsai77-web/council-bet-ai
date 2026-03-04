import asyncio
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# Initialize Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

async def get_gemini_2_5_flash(fixture):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Analyze {fixture} with live news. Predict winner.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        return res.text
    except Exception as e: return f"Gemini Error: {str(e)[:50]}"

async def get_groq_qwen3(data):
    try:
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        res = await client.chat.completions.create(model="qwen-3-32b", messages=[{"role":"user", "content":data}])
        return res.choices[0].message.content
    except Exception as e: return f"Groq Error: {str(e)[:50]}"

async def get_openrouter_glm45(data):
    try:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        res = await client.chat.completions.create(model="z-ai/glm-4.5-air:free", messages=[{"role":"user", "content":data}])
        return res.choices[0].message.content
    except Exception as e: return f"OpenRouter Error: {str(e)[:50]}"

async def get_hf_deepseek_r1(data):
    try:
        client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
        res = await client.chat.completions.create(model="deepseek-ai/DeepSeek-R1", messages=[{"role":"user", "content":data}])
        return res.choices[0].message.content
    except Exception as e: return f"DeepSeek Error: {str(e)[:50]}"

async def get_hf_kimi_k2(data):
    try:
        client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
        res = await client.chat.completions.create(model="moonshotai/Kimi-K2-Thinking", messages=[{"role":"user", "content":data}])
        return res.choices[0].message.content
    except Exception as e: return f"Kimi Error: {str(e)[:50]}"

async def main():
    game = "Arsenal vs Man City"
    print(f"📡 Consulting the Grand Council for: {game}...")

    # Parallel execution with error tolerance
    results = await asyncio.gather(
        get_gemini_2_5_flash(game),
        get_groq_qwen3(game),
        get_openrouter_glm45(game),
        get_hf_deepseek_r1(game),
        get_hf_kimi_k2(game),
        return_exceptions=True
    )

    # Prepare payload - Keys must match SQL column names exactly!
    payload = {
        "game_id": game,
        "gemini_2_5": results[0],
        "qwen_3": results[1],
        "glm_4_5": results[2],
        "deepseek_r1": results[3],
        "kimi_k2": results[4]
    }

    try:
        supabase.table("council_picks").insert(payload).execute()
        print("✅ Success! All 5 Council Members have recorded their votes in Supabase.")
    except Exception as e:
        print(f"❌ Supabase Insert Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
