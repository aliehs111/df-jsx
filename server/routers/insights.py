# server/routers/insights.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/insights", tags=["insights"])

class InsightRequest(BaseModel):
    text: str  # Later you’ll replace this with real dataset reference or content

class InsightResponse(BaseModel):
    insights: List[str]

@router.post("/", response_model=InsightResponse)
async def generate_insights(request: InsightRequest):
    # TODO: Replace with real model inference or API call
    dummy_insights = [
        "'Age' has 12% missing values.",
        "'Income' is skewed with outliers over $1M.",
        "'Has_Debt' is correlated with 'Target' (r = 0.52).",
        "Suggested: impute 'Age', log-transform 'Income', encode 'Has_Debt'."
    ]
    return InsightResponse(insights=dummy_insights)
