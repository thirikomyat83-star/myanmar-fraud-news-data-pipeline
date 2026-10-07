# ==============================================================================
# CELL 2: FASTAPI SERVER, /health, HUGGING FACE MODEL & UI DASHBOARD
# ==============================================================================



import nest_asyncio
nest_asyncio.apply()

import boto3
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uvicorn
import threading
import os

app = FastAPI(title="NEX4 - AI Fraud News Detection Platform")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:4566")
BUCKET_NAME = os.getenv("BUCKET_NAME", "myanmar-news-lake")
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5444/news_warehouse")

embedder = None
hf_classifier = None

@app.on_event("startup")
def load_production_models():
    global embedder, hf_classifier
    print(">>> [API SERVER] Initializing Embedder & Hugging Face Model...")
    try:
        embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        model_id = "Poe255M/myanmar-fraud-detection-final"
        hf_classifier = pipeline("text-classification", model=model_id)
        print(">>> [API SERVER] ✅ Models loaded successfully.")
    except Exception as e:
        print(f">>> [API SERVER] ⚠️ Model loading error: {e}")

# /health Endpoint (Production Readiness Monitoring)
@app.get("/health")
def health_check():
    status = {"api": "ok", "database": "down", "model": "down", "storage": "down"}
    try:
        conn = psycopg2.connect(DB_URL)
        conn.close()
        status["database"] = "ok"
    except Exception:
        pass
        
    if hf_classifier is not None and embedder is not None:
        status["model"] = "ready"
        
    try:
        s3_client = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id="test", aws_secret_access_key="test", region_name="ap-southeast-1")
        s3_client.list_buckets()
        status["storage"] = "ok"
    except Exception:
        pass
    return status

@app.get("/api/metrics")
def get_metrics():
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), SUM(is_fraud) FROM curated_news;")
            total, fraud = cur.fetchone()
        conn.close()
        return {"total": total or 0, "fraud": int(fraud or 0), "real": (total or 0) - int(fraud or 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

class SearchQuery(BaseModel):
    text: str

@app.post("/api/semantic-search")
def semantic_search(query: SearchQuery):
    try:
        vec = embedder.encode([query.text])[0].tolist()
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT news_text, category, is_fraud, 1 - (embedding <=> %s::vector) AS score
                FROM curated_news 
                ORDER BY embedding <=> %s::vector 
                LIMIT 3;
            """, (vec, vec))
            results = [{"text": r[0][:200]+"...", "category": r[1], "is_fraud": r[2], "similarity": round(r[3]*100, 2)} for r in cur.fetchall()]
        conn.close()
        return {"query": query.text, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/predict")
def predict_fraud(query: SearchQuery):
    if hf_classifier is None:
        raise HTTPException(status_code=503, detail="Fine-tuned model not loaded")
    try:
        res = hf_classifier(query.text)[0]
        label = res['label']
        score = round(res['score'] * 100, 2)
        
        model_config = getattr(hf_classifier.model.config, "id2label", None)
        if model_config and label in model_config:
            actual_label_name = model_config[int(label.replace("LABEL_", "")) if "LABEL_" in label else 0]
        else:
            actual_label_name = label
            
        is_fraud = True if str(actual_label_name).lower() in ['label_1', 'fraud', '1', 'positive', 'fraudulent'] else False
        
        return {
            "prediction": "Fraud" if is_fraud else "Real", 
            "confidence": score,
            "model": "Poe255M/myanmar-fraud-detection-final"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NEX4 | AI Data Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-slate-100 text-slate-800 min-h-screen flex flex-col p-8">
        <div class="max-w-4xl mx-auto w-full">
            <h1 class="text-3xl font-extrabold text-slate-900 mb-2">Myanmar Fraud News AI Platform</h1>
            <p class="text-slate-500 mb-6">AI Data Pipeline • Local Cloud Simulation (Cloud-Ready)</p>
            <div class="bg-white p-6 rounded-xl shadow-md mb-6" id="stats">Loading metrics...</div>
            <div class="bg-white p-6 rounded-xl shadow-md">
                <textarea id="q" rows="3" class="w-full border p-3 rounded-lg mb-4" placeholder="သတင်းစာသားတစ်ခု ရိုက်ထည့်ပါ..."></textarea>
                <button onclick="search()" class="bg-blue-600 text-white font-bold px-6 py-2 rounded-lg">Analyze with AI</button>
                <div id="res" class="mt-6 hidden"></div>
            </div>
        </div>
        <script>
            fetch('/api/metrics').then(r=>r.json()).then(d=>{
                document.getElementById('stats').innerHTML = `<div class="flex justify-around"><div>Total: <b>${d.total}</b></div><div>Fraud: <b class="text-red-600">${d.fraud}</b></div><div>Real: <b class="text-green-600">${d.real}</b></div></div>`;
            });
            async function search() {
                const text = document.getElementById('q').value;
                if(!text) return;
                const [sRes, pRes] = await Promise.all([
                    fetch('/api/semantic-search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})}).then(r=>r.json()),
                    fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})}).then(r=>r.json())
                ]);
                let resDiv = document.getElementById('res');
                resDiv.classList.remove('hidden');
                resDiv.innerHTML = `<div class="p-4 rounded-lg ${pRes.prediction === 'Fraud' ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'} mb-4"><b>Prediction:</b> ${pRes.prediction} (${pRes.confidence}%)</div>`;
            }
        </script>
    </body>
    </html>
    """

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("\n>>> 🚀 FastAPI Server အောင်မြင်စွာ တက်လာပါပြီ!")
print(">>> 🌐 Dashboard Link: http://localhost:8000/dashboard")
print(">>> 🩺 Health Check: http://localhost:8000/health\n")