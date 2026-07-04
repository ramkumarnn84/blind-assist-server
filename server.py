"""
Blind Assist Server - Cloud Edition (Render.com)
- Live streaming + guide controls
- Scene Analysis with Shapley attribution + Explainable AI + GPT-4o advisory
"""
import asyncio, json, os, time, base64
from datetime import datetime
from collections import deque
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import numpy as np

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

app = FastAPI()
phone_connection = None
guide_connections = []
connection_status = {"phone": False}

frame_counter = 0
last_gpt_analysis_time = 0

# 3D scene connections
scene_3d_connections = []

knowledge_base = {
    "decisions": deque(maxlen=200),
    "shapley_log": deque(maxlen=100),
    "communication_log": deque(maxlen=100),
    "gpt_advisories": deque(maxlen=50),
    "latest_frame": None
}

# ===== SHAPLEY ATTRIBUTION =====
class ShapleyAttributor:
    FEATURES = ['distance', 'position', 'motion', 'object_type', 'speed', 'box_size']

    def compute_shapley(self, detection):
        fv = {
            'distance': self._distance_score(detection.get('distance', 'far')),
            'position': self._position_score(detection.get('direction', 'left')),
            'motion': self._motion_score(detection.get('motion', 'stationary')),
            'object_type': self._object_score(detection.get('label', 'unknown')),
            'speed': self._speed_score(detection.get('motion', 'stationary')),
            'box_size': detection.get('box_area', 0.1)
        }
        total = sum(fv.values())
        if total == 0: total = 1
        shapley_vals = {k: round(v / total, 3) for k, v in fv.items()}
        top_feature = max(shapley_vals, key=shapley_vals.get)
        explanation = self._explain(top_feature, detection)
        return {
            "features": fv,
            "shapley_values": shapley_vals,
            "top_contributor": top_feature,
            "explanation": explanation,
            "risk_score": round(total / 6, 3)
        }

    def _distance_score(self, dist):
        if '<1m' in str(dist): return 1.0
        if '1' in str(dist): return 0.7
        if '2' in str(dist): return 0.5
        if '3' in str(dist): return 0.3
        return 0.1

    def _position_score(self, direction):
        return 1.0 if direction == 'center' else 0.4

    def _motion_score(self, motion):
        if 'approach' in str(motion): return 1.0
        if 'moving' in str(motion): return 0.4
        return 0.1

    def _object_score(self, label):
        dangerous = {'person': 0.7, 'car': 1.0, 'motorcycle': 0.9, 'bicycle': 0.8, 'dog': 0.6, 'truck': 1.0, 'bus': 1.0}
        return dangerous.get(label, 0.3)

    def _speed_score(self, motion):
        if 'fast' in str(motion): return 1.0
        if 'approach' in str(motion): return 0.7
        return 0.1

    def _explain(self, top_feature, detection):
        explanations = {
            'distance': f"Primary risk factor: object is {detection.get('distance','unknown')} away",
            'position': f"Object is {detection.get('direction','unknown')} - directly in walking path" if detection.get('direction')=='center' else f"Object on {detection.get('direction','side')}",
            'motion': f"Object is {detection.get('motion','stationary')} - movement pattern is key concern",
            'object_type': f"{detection.get('label','Object')} type inherently requires caution",
            'speed': f"Speed of movement is the primary danger factor",
            'box_size': f"Object appears large in frame indicating proximity"
        }
        return explanations.get(top_feature, "Multiple factors contributed to this decision")

shapley = ShapleyAttributor()

# ===== COMMUNICATION TRACKER =====
class CommunicationTracker:
    def __init__(self):
        self.warnings = deque(maxlen=50)
        self.patterns = []

    def log_warning(self, warning_type, direction, timestamp):
        self.warnings.append({"type": warning_type, "direction": direction, "time": timestamp})
        self._analyze_patterns()

    def _analyze_patterns(self):
        self.patterns = []
        if len(self.warnings) < 3: return
        recent = list(self.warnings)[-10:]
        if len(recent) >= 3:
            last3 = [w['type'] for w in recent[-3:]]
            if len(set(last3)) == 1:
                self.patterns.append({"issue": "REPETITIVE", "detail": f"Same warning '{last3[0]}' repeated 3+ times"})
        if len(recent) >= 2:
            dirs = [w['direction'] for w in recent[-3:] if w['direction']]
            if 'left' in dirs and 'right' in dirs:
                self.patterns.append({"issue": "CONFLICTING", "detail": "Both LEFT and RIGHT suggested within short period"})
        if len(recent) >= 5:
            time_span = recent[-1]['time'] - recent[-5]['time']
            if time_span < 10:
                self.patterns.append({"issue": "OVERLOAD", "detail": f"5 warnings in {time_span:.0f}s - information overload"})

    def get_analysis(self):
        return {
            "total_warnings": len(self.warnings),
            "recent_warnings": list(self.warnings)[-5:],
            "patterns": self.patterns,
            "recommendation": self._recommend()
        }

    def _recommend(self):
        if any(p['issue'] == 'CONFLICTING' for p in self.patterns):
            return "CRITICAL: Stop giving directions. Wait for scene to stabilize."
        if any(p['issue'] == 'OVERLOAD' for p in self.patterns):
            return "Reduce warning frequency. Only announce highest-priority threat."
        if any(p['issue'] == 'REPETITIVE' for p in self.patterns):
            return "Vary the message or reduce frequency."
        return "Communication pattern is healthy."

comm_tracker = CommunicationTracker()

def log_decision(detection, shapley_result, risk_level):
    entry = {
        "timestamp": time.time(),
        "time_str": datetime.now().strftime("%H:%M:%S"),
        "detection": detection,
        "shapley": shapley_result,
        "risk_level": risk_level,
    }
    knowledge_base["decisions"].append(entry)
    knowledge_base["shapley_log"].append(shapley_result)
    if risk_level in ['critical', 'high', 'medium']:
        comm_tracker.log_warning(risk_level, detection.get('direction', ''), time.time())
        knowledge_base["communication_log"].append({
            "time": entry["time_str"], "type": risk_level,
            "object": detection.get('label', ''), "direction": detection.get('direction', ''),
            "distance": detection.get('distance', '')
        })


# ===== GPT-4o ADVISORY =====
guide_gpt_connections = []

async def get_gpt_advisory(frame_base64=None):
    global frame_counter
    if not OPENAI_API_KEY:
        return "No OpenAI API key configured"

    recent_decisions = list(knowledge_base["decisions"])[-5:]
    comm_analysis = comm_tracker.get_analysis()

    context = """You are an accessibility AI for blind navigation. Analyze this camera frame.

TASK 1 - SCENE RECONSTRUCTION:
List EVERY visible object with precise spatial info as JSON array.

TASK 2 - PATH:
{"direction":"left/center/right","width":"narrow/medium/wide","description":"path description"}

TASK 3 - ADVISORY (1 sentence):
Was the system warning appropriate?

Detection data: """ + json.dumps(recent_decisions[-3:], default=str) + """
Communication: """ + json.dumps(comm_analysis, default=str) + """

RESPOND EXACTLY:
OBJECTS: [json array]
PATH: {json object}
ADVISORY: text"""

    messages = [{"role": "system", "content": "You are a spatial mapping AI for blind navigation assistance. Describe the physical environment layout for safe navigation."}]

    if frame_base64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": context},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"}}
        ]})
    else:
        messages.append({"role": "user", "content": context})

    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "messages": messages, "max_tokens": 800, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                full_response += delta
                                for g in guide_gpt_connections:
                                    try: await g.send_json({"type": "gpt_chunk", "text": delta})
                                    except: pass
                        except: pass

        for g in guide_gpt_connections:
            try: await g.send_json({"type": "gpt_done", "full_text": full_response})
            except: pass

        knowledge_base["gpt_advisories"].append({"time": datetime.now().strftime("%H:%M:%S"), "advisory": full_response})
        return full_response
    except Exception as e:
        err = f"GPT Error: {str(e)}"
        for g in guide_gpt_connections:
            try: await g.send_json({"type": "gpt_error", "text": err})
            except: pass
        return err


async def auto_analyze_frame(detections, frame_base64):
    global last_gpt_analysis_time, frame_counter
    now = time.time()
    frame_counter += 1

    if now - last_gpt_analysis_time >= 10 and len(detections) > 0 and len(guide_gpt_connections) > 0:
        last_gpt_analysis_time = now
        asyncio.create_task(get_gpt_advisory(frame_base64))

# ===== API ENDPOINTS =====
@app.post("/api/analyze")
async def analyze_scene(request: Request):
    frame = knowledge_base.get("latest_frame")
    advisory = await get_gpt_advisory(frame)
    return JSONResponse({"advisory": advisory, "communication": comm_tracker.get_analysis()})

@app.get("/api/shapley")
async def get_shapley_data():
    return JSONResponse({
        "recent": list(knowledge_base["shapley_log"])[-10:],
        "decisions": list(knowledge_base["decisions"])[-10:],
        "communication": comm_tracker.get_analysis()
    })

@app.get("/api/knowledge")
async def get_knowledge():
    return JSONResponse({
        "decisions": list(knowledge_base["decisions"])[-20:],
        "shapley": list(knowledge_base["shapley_log"])[-10:],
        "communication": comm_tracker.get_analysis(),
        "advisories": list(knowledge_base["gpt_advisories"])[-5:]
    })

@app.get("/api/corrections")
async def get_corrections():
    advisories = list(knowledge_base["gpt_advisories"])[-3:]
    corrections = [{"text": a.get("advisory", ""), "time": a.get("time", "")} for a in advisories]
    return JSONResponse({"corrections": corrections, "patterns": comm_tracker.get_analysis().get("patterns", [])})

@app.websocket("/ws/gpt-stream")
async def gpt_stream_ws(websocket: WebSocket):
    await websocket.accept()
    guide_gpt_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        guide_gpt_connections.remove(websocket)

@app.get("/status")
async def status():
    return {"phone_connected": connection_status["phone"], "guide_count": len(guide_connections)}


# ===== HTML TEMPLATES =====
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def load_template(filename):
    filepath = os.path.join(TEMPLATE_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def guide_page():
    return HTMLResponse(content=load_template("guide.html"))

@app.get("/phone", response_class=HTMLResponse)
async def phone_page():
    return HTMLResponse(content=load_template("phone.html"))

# ===== WEBSOCKETS =====
@app.websocket("/ws/phone")
async def phone_ws(websocket: WebSocket):
    global phone_connection
    await websocket.accept()
    phone_connection = websocket
    connection_status["phone"] = True
    for g in guide_connections:
        try: await g.send_json({"type": "phone_status", "data": {"connected": True}})
        except: pass
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg["type"] == "frame":
                knowledge_base["latest_frame"] = msg.get("data")
                for g in guide_connections:
                    try: await g.send_json(msg)
                    except: pass
            elif msg["type"] == "detections" and msg.get("data", {}).get("objects"):
                detections = msg["data"]["objects"]
                for obj in detections:
                    sv = shapley.compute_shapley(obj)
                    log_decision(obj, sv, obj.get("risk", "low"))
                await auto_analyze_frame(detections, knowledge_base.get("latest_frame"))
                for g in guide_connections:
                    try: await g.send_json(msg)
                    except: pass
    except WebSocketDisconnect:
        phone_connection = None
        connection_status["phone"] = False
        for g in guide_connections:
            try: await g.send_json({"type": "phone_status", "data": {"connected": False}})
            except: pass

@app.websocket("/ws/guide")
async def guide_ws(websocket: WebSocket):
    await websocket.accept()
    guide_connections.append(websocket)
    await websocket.send_json({"type": "phone_status", "data": {"connected": connection_status["phone"]}})
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg["type"] == "guidance" and phone_connection:
                await phone_connection.send_json(msg)
    except WebSocketDisconnect:
        guide_connections.remove(websocket)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Server running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
