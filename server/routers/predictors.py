from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from server.routers.college_earnings_model import predict_college_earnings, load_artifacts as ce_load
ce_load()

import re, json
from pathlib import Path
import math


router = APIRouter(prefix="/predictors", tags=["predictors"])

from server.routers.college_earnings_model import predict_college_earnings, load_artifacts as ce_load
ce_load()



RULES_DIR = Path(__file__).parent / "rules"

class Params(BaseModel):
    text: str = Field(..., min_length=1)
    audience: str = Field(..., pattern=r"^(ESL|OlderAdults|General)$")
    medium: str = Field(..., pattern=r"^(SMS|Email)$")
    intent: Optional[str] = Field(None)


class ConfusionSource(BaseModel):
    type: str
    evidence: List[str] = []
    note: Optional[str] = None

class PredictResponse(BaseModel):
    status: str = "success"
    model: str = "accessibility_risk"
    version: str = "v1"
    misinterpretation_probability: float
    risk_bucket: str
    confusion_sources: List[ConfusionSource]
    rewrite_15_words: str
    warnings: List[str] = []
    
# --- models ---
class Overrides(BaseModel):
    density_thresholds: Optional[Dict[str, float]] = None
    audience_weights: Optional[Dict[str, float]] = None
    audience_soften: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_categories: Optional[List[str]] = None
    sensitivity: Optional[float] = None
    # NEW:
    category_weights: Optional[Dict[str, float]] = None  # keys: idioms,jargon,ambiguous_time,polysemy,numeracy_date
    sigmoid_center: Optional[float] = None               # default ~0.35–0.40
    sigmoid_k: Optional[float] = None                    # default 6.0




class GenericPredictRequest(BaseModel):
    model: str                                  # e.g., "accessibility_risk" or "college_earnings_v1_75k_5y"
    params: Dict[str, Any] = {}                 # free-form per model
    overrides: Optional[Overrides] = None       # used by accessibility_risk

class EarningsDriver(BaseModel):
    factor: str
    direction: str
    weight: float

class EarningsResponse(BaseModel):
    status: str = "success"
    model: str = "college_earnings"
    version: str
    probability: float
    risk_bucket: str
    drivers: List[EarningsDriver] = []
    confidence: Optional[str] = None
    warnings: List[str] = []



DEFAULT_RULES: Dict[str, Any] = {
    "idioms": ["asap", "heads up", "roll out", "touch base", "circle back", "in the loop"],
    "jargon": ["synergy", "leverage", "deliverable", "KPI", "OKR", "onboarding", "scope creep"],
    "ambiguous_time": ["soon", "shortly", "end of day", "next week", "later", "as soon as possible"],
    "polysemy": ["charge", "bill", "fine", "credit", "debit", "statement"],
    "numeracy_date": [
        r"\b\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d+%\b",
        r"\b\$\d+(?:,\d{3})*(?:\.\d+)?\b",
    ],
    "audience_weights": {"ESL": 1.25, "OlderAdults": 1.2, "General": 1.0},
    "density_thresholds": {"low": 0.25, "high": 0.65},
    "idiom_rewrites": {
        "asap": "as soon as possible",
        "roll out": "launch",
        "heads up": "note",
        "touch base": "talk",
        "circle back": "follow up",
        "in the loop": "informed",
    },
}

def load_rules() -> Dict[str, Any]:
    rules = DEFAULT_RULES.copy()
    if RULES_DIR.exists():
        for name in [
            "idioms", "jargon", "ambiguous_time", "polysemy", "numeracy_date",
            "audience_weights", "density_thresholds", "idiom_rewrites",
        ]:
            p = RULES_DIR / f"{name}.json"
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        rules[name] = json.load(f)
                except Exception:
                    pass
    return rules

RULES = load_rules()
_word = re.compile(r"[\w']+")

def tokenize(text: str) -> List[str]:
    return _word.findall(text.lower())

def find_terms(text: str, terms: List[str]) -> List[str]:
    hits = []
    for t in terms:
        if re.search(rf"\b{re.escape(t.lower())}\b", text.lower()):
            hits.append(t)
    return hits

def find_regex(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pat in patterns:
        try:
            if re.search(pat, text):
                hits.append(pat)
        except re.error:
            continue
    return hits

def bucket_from_prob(p: float, thresholds: Dict[str, float]) -> str:
    if p >= thresholds.get("high", 0.65):
        return "High"
    if p >= thresholds.get("low", 0.25):
        return "Medium"
    return "Low"

def rewrite_15(text: str, idiom_map: Dict[str, str]) -> str:
    out = text
    for k in sorted(idiom_map.keys(), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(k)}\b", idiom_map[k], out, flags=re.IGNORECASE)
    words = tokenize(out)
    return " ".join(words[:15])


    
def score_accessibility(params: Params, overrides: Optional[Overrides] = None) -> PredictResponse:
    rules = RULES  # defaults loaded at startup

    # Which categories are active for this run
    enabled = set(overrides.enable_categories) if (overrides and overrides.enable_categories) else {
        "idioms", "jargon", "ambiguous_time", "polysemy", "numeracy_date"
    }

    text = params.text.strip()
    tokens = tokenize(text)
    n_tokens = max(1, len(tokens))

    # Term matches (respect enabled categories)
    idioms = find_terms(text, rules["idioms"]) if "idioms" in enabled else []
    jargon = find_terms(text, rules["jargon"]) if "jargon" in enabled else []
    ambig  = find_terms(text, rules["ambiguous_time"]) if "ambiguous_time" in enabled else []
    poly   = find_terms(text, rules["polysemy"]) if "polysemy" in enabled else []
    nums   = find_regex(text, rules["numeracy_date"]) if "numeracy_date" in enabled else []

    # Build confusion sources list (only include non-empty)
    confusion: List[ConfusionSource] = []
    if idioms:
        confusion.append(ConfusionSource(type="Idioms/Colloquialisms", evidence=idioms, note="Replace with plain words"))
    if jargon:
        confusion.append(ConfusionSource(type="Jargon/Corporate", evidence=jargon, note="Use everyday language"))
    if ambig:
        confusion.append(ConfusionSource(type="Ambiguous Time", evidence=ambig, note="Give exact dates/times"))
    if poly:
        confusion.append(ConfusionSource(type="Polysemous Words", evidence=poly, note="Clarify meaning"))
    if nums:
        confusion.append(ConfusionSource(type="Numbers/Dates", evidence=nums, note="Spell out or format clearly"))

    # ---- density with adjustable category weights ----
    default_cw = {"idioms": 0.35, "jargon": 0.35, "ambiguous_time": 0.30, "polysemy": 0.10, "numeracy_date": 0.10}
    cw = (overrides.category_weights if (overrides and overrides.category_weights) else default_cw)

    density = (
        (cw.get("idioms", 0)         * len(idioms)) +
        (cw.get("jargon", 0)         * len(jargon)) +
        (cw.get("ambiguous_time", 0) * len(ambig)) +
        (cw.get("polysemy", 0)       * len(poly)) +
        (cw.get("numeracy_date", 0)  * len(nums))
    ) / n_tokens

    sensitivity = overrides.sensitivity if (overrides and overrides.sensitivity is not None) else 1.0
    density *= sensitivity

    # ---- sigmoid with adjustable center/slope ----
    center = overrides.sigmoid_center if (overrides and overrides.sigmoid_center is not None) else 0.35
    k = overrides.sigmoid_k if (overrides and overrides.sigmoid_k is not None) else 6.0
    base = 1.0 / (1.0 + math.exp(-k * (density - center)))
    base = min(0.9, max(0.0, base))

    # ---- audience weighting ----
    audience_weights = (overrides.audience_weights or rules["audience_weights"]) if overrides else rules["audience_weights"]
    w = audience_weights.get(params.audience, 1.0)
    soften = overrides.audience_soften if (overrides and overrides.audience_soften is not None) else 0.6
    if w > 1.0:
        w = 1.0 + (w - 1.0) * soften

    prob = max(0.0, min(1.0, base * w))

    # ---- thresholds and bucket ----
    thresholds = (overrides.density_thresholds or rules["density_thresholds"]) if overrides else rules["density_thresholds"]
    bucket = bucket_from_prob(prob, thresholds)

    # ---- ≤15-word rewrite ----
    rewrite = rewrite_15(text, rules.get("idiom_rewrites", {}))

    return PredictResponse(
        misinterpretation_probability=prob,
        risk_bucket=bucket,
        confusion_sources=confusion,
        rewrite_15_words=rewrite,
        warnings=[],
    )




@router.post(
    "/infer",
    response_model=Union[PredictResponse, EarningsResponse],
    tags=["predictors"]
)
async def infer(req: GenericPredictRequest):
    if req.model == "accessibility_risk":
        # params must match the Params schema
        p = Params(**req.params)
        return score_accessibility(p, req.overrides)

    if req.model == "college_earnings_v1_75k_5y":
        return predict_college_earnings(req.params or {})

    raise HTTPException(status_code=400, detail="Unsupported model")



