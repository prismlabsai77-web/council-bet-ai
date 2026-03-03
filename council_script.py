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

async def get_gemini_2_flash(fixture):
    """Gemini 2.0 with Search Grounding for real-time news."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"Find today's injuries and weather for {fixture}. Predict winner."
    # 2026 Search Grounding Config
    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    return res.text

async def get_groq_qwen3(data):
    """Qwen 3 32B on Groq for sub-second analysis."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    res = await client.chat.completions.create(
        model="qwen-3-32b", 
        messages=[{"role": "user", "content": f"Analyze stats: {data}. Return JSON prediction."}]
    )
    return res.choices[0].message.content

async def get_openrouter_glm45(data):
    """GLM 4.5 Air for agentic reasoning."""
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    res = await client.chat.completions.create(
        model="z-ai/glm-4.5-air:free",
        messages=[{"role": "user", "content": f"Reason about {data}. Prediction?"}]
    )
    return res.choices[0].message.content

async def get_hf_deepseek_r1(data):
    """DeepSeek R1 on HF for logical auditing."""
    client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
    res = await client.chat.completions.create(model="deepseek-ai/DeepSeek-R1", messages=[{"role": "user", "content": data}])
    return res.choices[0].message.content

async def get_hf_kimi_k2(data):
    """Kimi K2 Thinking on HF for deep simulation."""
    client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
    res = await client.chat.completions.create(model="moonshotai/Kimi-K2-Thinking", messages=[{"role": "user", "content": data}])
    return res.choices[0].message.content

async def main():
    game = "Arsenal vs Man City"
    # Execute all 5 models in parallel
    results = await asyncio.gather(
        get_gemini_2_flash(game),
        get_groq_qwen3(game),
        get_openrouter_glm45(game),
        get_hf_deepseek_r1(game),
        get_hf_kimi_k2(game)
    )

    # Insert into Supabase
    supabase.table("council_picks").insert({
        "game_id": game,
        "gemini_2_0": results[0],
        "qwen_3": results[1],
        "glm_4_5": results[2],
        "deepseek_r1": results[3],
        "kimi_k2": results[4]
    }).execute()
    print("🚀 All 5 Council Members have voted!")

if __name__ == "__main__":
    asyncio.run(main())
