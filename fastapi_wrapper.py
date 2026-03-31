from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from call_groq import call_groq

app = FastAPI(title="Hospitality Routing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RoutingRequest(BaseModel):
    query: str

class RoutingResponse(BaseModel):
    specialist: str
    answer: str
    elapsed_time: float

@app.post("/route", response_model=RoutingResponse)
async def route_endpoint(request: RoutingRequest):
    start_time = time.time()

    # Step 1: Classify the query into one specialist
    classify_prompt = f"""You are a hospitality query classifier. Read the user's request and decide which specialist should handle it.

Options:
- budget: for questions about price, affordable hotels, budget ranges
- location: for questions about best areas, neighbourhoods, proximity to attractions
- amenities: for questions about facilities (pool, gym, breakfast, spa, beachfront, etc.)
- dining: for questions about restaurants, cuisine, food recommendations

Query: {request.query}

Output ONLY one word: budget, location, amenities, or dining.
"""
    specialist = call_groq(classify_prompt, node_name="CLASSIFIER").strip().lower()

    # Step 2: Call the corresponding specialist
    if specialist == "budget":
        prompt = f"""You are a budget specialist for hospitality. Answer the following query with budget‑focused advice, mentioning approximate price ranges and value for money.

Query: {request.query}"""
    elif specialist == "location":
        prompt = f"""You are a location specialist for hospitality. Answer the following query with location‑focused advice, suggesting areas, neighbourhoods, or proximity to attractions.

Query: {request.query}"""
    elif specialist == "amenities":
        prompt = f"""You are an amenities specialist for hospitality. Answer the following query focusing on facilities like pools, gyms, breakfast, spas, beachfront, etc.

Query: {request.query}"""
    elif specialist == "dining":
        prompt = f"""You are a dining specialist for hospitality. Answer the following query with restaurant or cuisine recommendations.

Query: {request.query}"""
    else:
        specialist = "general"
        prompt = f"""You are a general hospitality advisor. Answer the following query.

Query: {request.query}"""

    answer = call_groq(prompt, node_name="SPECIALIST")
    elapsed = time.time() - start_time
    return RoutingResponse(specialist=specialist, answer=answer, elapsed_time=elapsed)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)