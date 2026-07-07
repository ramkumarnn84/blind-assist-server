"""
Blind Assist V2 — Meta AI Powered Immersive Navigation
Uses Meta Llama 4 Maverick (via Together.ai) for vision-based trajectory guidance.
Provides continuous spatial narration, not just obstacle warnings.
"""
import asyncio, json, os, time, base64
from datetime import datetime
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx

# Meta Llama via Together.ai (free tier, no credit card)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
# Fallback to OpenAI (set via environment variable on Render)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

app = FastAPI(title="Blind Assist V2 - Meta AI Navigation")

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def load_template(filename):
    with open(os.path.join(TEMPLATE_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()

# ===== META LLAMA VISION (via Together.ai) =====
async def call_meta_llama_vision(frame_base64: str, prompt: str) -> str:
    """Call Meta Llama 4 Maverick vision model via Together.ai"""
    if TOGETHER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}}
                        ]}],
                        "max_tokens": 500
                    }
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            pass  # Fall through to OpenAI

    # Fallback: OpenAI GPT-4o-mini
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"}}
                        ]}],
                        "max_tokens": 500
                    }
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    return f"Error: {resp.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    return "No AI API configured."

# ===== NAVIGATION PROMPTS =====

TRAJECTORY_PROMPT = """You are a real-time navigation AI for a blind person walking with their phone camera.

Your job is NOT to warn about danger — it is to GUIDE them through the space like a personal navigator.

Analyze this camera frame and provide:

1. TRAJECTORY: Where exactly to walk next (be specific: "Walk straight 3 meters", "Turn slightly left", "Take 2 steps forward then turn right")
2. SURROUNDINGS: Brief spatial layout (what's on left, right, ahead — surfaces, walls, openings)
3. PATH QUALITY: Floor condition, slope, steps, width of path
4. LANDMARKS: Doors, signs, furniture, intersections that help orientation

Rules:
- Speak like a GPS navigator, not an alarm system
- Use simple directions: "on your left", "on your right", "straight ahead", "slightly left", "slightly right"
- Give DISTANCE in steps or meters
- Only mention obstacles as part of routing ("go around the chair on your left"), not as warnings
- Keep it to 3-4 short sentences max
- Start with the immediate next action
- NEVER use clock directions (no "2 o'clock", "10 o'clock" etc.)

Example good response:
"Walk forward 4 meters, path is clear. Door on your right, about 3 meters ahead. Smooth tile floor. Wall running along your left side."

Example BAD response (don't do this):
"Warning! Chair detected! Danger ahead!"
"Object at 2 o'clock"
"""

SCENE_PROMPT = """You are describing a scene to a blind person. Give a rich, spatial description:
- What room/space is this? (indoor/outdoor, type)
- What's at each position? (left, center, right, near, far)
- What's the floor/ground like?
- Any text, signs, or labels visible?
- Colors, lighting, atmosphere
- People present (count, position, not identity)

Be vivid but concise. Help them build a mental map. 3-5 sentences."""

OCR_PROMPT = """Read ALL text visible in this image. Return ONLY the text content.
If it's a sign, label, screen, document, menu — read it completely.
If multiple text areas, separate with periods.
If no text visible, say "No text detected."
"""

IDENTIFY_PROMPT = """A blind person points their camera and asks "What is this?"
Identify the main object:
- What it is (specific: brand, type, color, size)
- Any text on it
- One sentence of useful context
Example: "Red Campbell's tomato soup can, 10 oz. Expiry date reads March 2027."
"""

CROSSWALK_PROMPT = """A blind person needs to cross a road. Analyze for safety:
1. Traffic light: red/green/yellow/none
2. Safe to cross: yes/no/wait
3. Vehicles: approaching/stopped/none, which direction
4. Crosswalk: visible/not visible
5. Action: one clear instruction

Be direct. Example: "Red light. Wait. Cars moving left to right. Crosswalk is directly ahead. Wait for green."
"""

# ===== API ENDPOINTS =====

@app.post("/api/navigate")
async def navigate(request: Request):
    """Main trajectory guidance — tells user WHERE to walk."""
    body = await request.json()
    frame = body.get("frame", "")
    if not frame:
        return JSONResponse({"guidance": "No frame received."})
    result = await call_meta_llama_vision(frame, TRAJECTORY_PROMPT)
    return JSONResponse({"guidance": result})

@app.post("/api/scene")
async def scene(request: Request):
    """Full scene description."""
    body = await request.json()
    frame = body.get("frame", "")
    if not frame:
        return JSONResponse({"description": "No frame received."})
    result = await call_meta_llama_vision(frame, SCENE_PROMPT)
    return JSONResponse({"description": result})

@app.post("/api/ocr")
async def ocr(request: Request):
    """Read text from image."""
    body = await request.json()
    frame = body.get("frame", "")
    if not frame:
        return JSONResponse({"text": "No frame received."})
    result = await call_meta_llama_vision(frame, OCR_PROMPT)
    return JSONResponse({"text": result})

@app.post("/api/identify")
async def identify(request: Request):
    """Identify object."""
    body = await request.json()
    frame = body.get("frame", "")
    if not frame:
        return JSONResponse({"description": "No frame received."})
    result = await call_meta_llama_vision(frame, IDENTIFY_PROMPT)
    return JSONResponse({"description": result})

@app.post("/api/crosswalk")
async def crosswalk(request: Request):
    """Crosswalk/road safety analysis."""
    body = await request.json()
    frame = body.get("frame", "")
    if not frame:
        return JSONResponse({"description": "No frame received."})
    result = await call_meta_llama_vision(frame, CROSSWALK_PROMPT)
    return JSONResponse({"description": result})

# ===== PAGES =====
@app.get("/", response_class=HTMLResponse)
async def phone_page():
    return HTMLResponse(content=load_template("phone.html"))

@app.get("/status")
async def status():
    return {
        "status": "running",
        "version": "2.0-meta-ai",
        "meta_llama": bool(TOGETHER_API_KEY),
        "openai_fallback": bool(OPENAI_API_KEY)
    }

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9091))
    ssl_cert = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BlindAssistApp", "server", "cert.pem")
    ssl_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BlindAssistApp", "server", "key.pem")

    print("=" * 50)
    print("  Blind Assist V2 — Meta AI Navigation")
    print("=" * 50)
    print(f"  Phone: https://192.168.1.10:{port}/")
    print("=" * 50)

    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
