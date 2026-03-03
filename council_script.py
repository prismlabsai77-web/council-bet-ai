import asyncio
import os
import json
from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI
from supabase import create_client

# Initialize Clients
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

async def get_gemini_2_5_flash(fixture):
    """Gemini 2.5 Flash with Search Grounding for live team news."""
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = f"Provide a detailed betting analysis for {fixture} using today's team news, injuries, and weather. Predict the winner."
        
        # 2026 Stable Gemini 2.5 Config
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return res.text
    except Exception as e:
        print(f"⚠️ Gemini 2.5 Error: {e}")
        return "Gemini Abstains (Quota/Limit)"

async def get_groq_qwen3(data):
    """Qwen 3 32B on Groq for ultra-fast trend analysis."""
    try:
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        res = await client.chat.completions.create(
            model="qwen-3-32b", 
            messages=[{"role": "user", "content": f"Analyze stats: {data}. Return JSON prediction."}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

async def get_openrouter_glm45(data):
    """GLM 4.5 Air for agentic reasoning."""
    try:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        res = await client.chat.completions.create(
            model="z-ai/glm-4.5-air:free",
            messages=[{"role": "user", "content": f"Reason about {data}. Prediction?"}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"OpenRouter Error: {e}"

async def get_hf_deepseek_r1(data):
    """DeepSeek R1 on HF for logical auditing."""
    try:
        client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
        res = await client.chat.completions.create(model="deepseek-ai/DeepSeek-R1", messages=[{"role": "user", "content": data}])
        return res.choices[0].message.content
    except Exception as e:
        return f"HF DeepSeek Error: {e}"

async def get_hf_kimi_k2(data):
    """Kimi K2 Thinking on HF for deep simulation."""
    try:
        client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
        res = await client.chat.completions.create(model="moonshotai/Kimi-K2-Thinking", messages=[{"role": "user", "content": data}])
        return res.choices[0].message.content
    except Exception as e:
        return f"HF Kimi Error: {e}"

async def main():
    game = "Arsenal vs Man City"
    
    # Execute all 5 models in parallel
    results = await asyncio.gather(
        get_gemini_2_5_flash(game),
        get_groq_qwen3(game),
        get_openrouter_glm45(game),
        get_hf_deepseek_r1(game),
        get_hf_kimi_k2(game)
    )

    # Insert into Supabase
    supabase.table("council_picks").insert({
        "game_id": game,
        "gemini_2_5": results[0],
        "qwen_3": results[1],
        "glm_4_5": results[2],
        "deepseek_r1": results[3],
        "kimi_k2": results[4],
        "created_at": "now()"
    }).execute()
    
    print("🚀 Grand Council 2026: Votes Recorded!")

if __name__ == "__main__":
    asyncio.run(main())
