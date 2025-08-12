# server/routers/college_earnings_model.py
# CPU-only hierarchical logistic inference for College Scorecard predictor
# 
# Predicts: P(earnings >= $75k at 5 years) using fixed effects + random effects (CIP, State)
# Artifacts are exported offline (from your training notebook) and loaded here at startup.
#
# Folder layout expected (local or mounted from S3 at deploy time):
#   server/routers/models/college_earnings/v1_75k_5y/
#       metadata.json
#       encoders.json
#       fixed_effects.json
#       random_cip.json
#       random_state.json
#       calibration.json
#       thresholds.json
#
# Minimal encoders.json example structure:
# {
#   "degree_level_vocab": ["Associate","Bachelor","Master","Professional","Doctoral"],
#   "state_vocab": ["CA","NY","TX", ...],
#   "cip4_vocab": ["1101","5203", ...],
#   "bins": {"program_size": [0, 50, 200, 1000, "inf"]}
# }
#
# Minimal fixed_effects.json example:
# { "intercept": -0.35,
#   "coefficients": {
#     "degree_level_Bachelor": 0.22,
#     "degree_level_Master": 0.51,
#     "degree_level_Professional": 0.78,
#     "degree_level_Doctoral": 0.66,
#     "public_private_Public": -0.04,
#     "public_private_Private": 0.04
#   }
# }

#
# Minimal random_cip.json example: { "1101": 0.20, "5203": -0.08, ... }
# Minimal random_state.json example: { "CA": 0.03, "NY": 0.05, ... }
# Minimal calibration.json (Platt): { "a": 1.02, "b": -0.03 }
# Minimal thresholds.json: { "low": 0.33, "high": 0.66 }

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import math
from pathlib import Path
import re

DEGREE_ALIASES = {
    "associate": "Associate", "associates": "Associate",
    "bachelor": "Bachelor", "bachelors": "Bachelor",
    "master": "Master", "masters": "Master",
    "professional": "Professional",
    "doctoral": "Doctoral", "doctorate": "Doctoral",
}
PRIVATE_ALIASES = {
    "public": "Public",
    "private": "Private",
    "private nonprofit": "Private", "private non-profit": "Private",
    "private for-profit": "Private",
}
def norm_degree(s: str) -> str:
    if not s: return ""
    k = re.sub(r"[^a-z]", "", s.lower())
    return DEGREE_ALIASES.get(k, s.strip())

def norm_public_private(s: str) -> str:
    if not s: return ""
    k = re.sub(r"\s+", " ", s.strip().lower())
    return PRIVATE_ALIASES.get(k, s.strip())

def norm_state(s: str) -> str:
    return (s or "").strip().upper()[:2]

def norm_cip4(s: str) -> str:
    return re.sub(r"\D", "", (s or ""))[:4]

# -----------------------------
# Globals (loaded at import)
# -----------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "college_earnings" / "v1_75k_5y"

_loaded = {
    "metadata": None,
    "encoders": None,
    "fixed": None,
    "rand_cip": None,
    "rand_state": None,
    "calib": None,
    "thresholds": None,
}

# -----------------------------
# Data models
# -----------------------------
@dataclass
class CollegeEarningsParams:
    cip4: str             # e.g., "1101" (Computer Science)
    degree_level: str     # One of encoders["degree_level_vocab"]
    state: str            # Two-letter state code (use vocab)
    public_private: Optional[str] = None  # "Public" | "Private" (optional for v1)

# -----------------------------
# Utilities
# -----------------------------

def _sigmoid(z: float) -> float:
    # Numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts(artifact_dir: Optional[Path] = None) -> None:
    """Load all artifacts into memory. Safe to call multiple times."""
    global ARTIFACT_DIR, _loaded
    if artifact_dir is not None:
        ARTIFACT_DIR = Path(artifact_dir)

    def try_load(name: str, fname: str, required: bool = True) -> Optional[Dict[str, Any]]:
        fp = ARTIFACT_DIR / fname
        if not fp.exists():
            if required:
                raise FileNotFoundError(f"Missing artifact: {fp}")
            return None
        return _load_json(fp)

    _loaded["metadata"] = try_load("metadata", "metadata.json")
    _loaded["encoders"] = try_load("encoders", "encoders.json")
    _loaded["fixed"] = try_load("fixed", "fixed_effects.json")
    _loaded["rand_cip"] = try_load("random_cip", "random_cip.json", required=False) or {}
    _loaded["rand_state"] = try_load("random_state", "random_state.json", required=False) or {}
    _loaded["calib"] = try_load("calib", "calibration.json", required=False) or {"a": 1.0, "b": 0.0}
    _loaded["thresholds"] = try_load("thresholds", "thresholds.json", required=False) or {"low": 0.33, "high": 0.66}


# Attempt load at import. If artifacts aren't present yet, you can call load_artifacts()
try:
    load_artifacts()
except Exception:
    # Defer failure until first prediction; this allows the app to boot before artifacts are added.
    pass


# -----------------------------
# Core inference
# -----------------------------

def _encode_fixed(params: CollegeEarningsParams) -> Tuple[Dict[str, float], List[str]]:
    enc = _loaded["encoders"] or {}
    coefs = _loaded["fixed"]["coefficients"] if _loaded["fixed"] else {}
    fixed_cols = set((enc.get("fixed_feature_columns") or coefs.keys()))
    feats: Dict[str, float] = {}
    warnings: List[str] = []

    # degree_level -> degree_level_<Label>
    if params.degree_level:
        key = f"degree_level_{params.degree_level}"
        if key in fixed_cols:
            feats[key] = 1.0
        else:
            warnings.append(f"Unknown degree_level '{params.degree_level}' — using baseline.")
    else:
        warnings.append("Missing degree_level — using baseline.")

    # public_private -> public_private_<Label>
    if params.public_private:
        key_pp = f"public_private_{params.public_private}"
        if key_pp in fixed_cols:
            feats[key_pp] = 1.0
        else:
            warnings.append(f"Unknown public_private '{params.public_private}' — ignoring.")

    # (If you later add size_bin, build key size_bin_<bin> here.)
    return feats, warnings



def _random_effects(cip4: str, state: str) -> Tuple[float, float, List[str]]:
    rc = _loaded["rand_cip"] or {}
    rs = _loaded["rand_state"] or {}
    u = rc.get(cip4, 0.0)
    v = rs.get(state, 0.0)
    warnings: List[str] = []
    if cip4 not in rc:
        warnings.append(f"No random effect for CIP '{cip4}' — using 0.")
    if state not in rs:
        warnings.append(f"No random effect for state '{state}' — using 0.")
    return u, v, warnings


def _apply_calibration(p_raw: float) -> float:
    calib = _loaded.get("calib") or {}
    # Support both {a,b} and {coef,intercept} shapes
    a = float(calib.get("a", calib.get("coef", 1.0)))
    b = float(calib.get("b", calib.get("intercept", 0.0)))
    # Platt scaling on log-odds
    eps = 1e-8
    pr = max(eps, min(1.0 - eps, p_raw))
    logit = math.log(pr / (1.0 - pr))
    return _sigmoid(a * logit + b)



def _bucket(p: float) -> str:
    th = _loaded.get("thresholds") or {"low": 0.33, "high": 0.66}
    if p >= th.get("high", 0.66):
        return "High"
    if p >= th.get("low", 0.33):
        return "Medium"
    return "Low"


def predict_college_earnings(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute probability and drivers using loaded artifacts.
    params expects keys: cip4, degree_level, state, optional public_private
    """
    # Ensure artifacts are loaded
    if not _loaded.get("fixed") or not _loaded.get("encoders"):
        load_artifacts()

    # normalize inputs
    p = CollegeEarningsParams(
        cip4=norm_cip4(params.get("cip4", "")),
        degree_level=norm_degree(params.get("degree_level", "")),
        state=norm_state(params.get("state", "")),
        public_private=(norm_public_private(params.get("public_private")) if params.get("public_private") else None),
    )

    intercept = float((_loaded["fixed"] or {}).get("intercept", 0.0))
    coefs = (_loaded["fixed"] or {}).get("coefficients", {}) or {}

    # Build fixed features
    x, warn1 = _encode_fixed(p)
    # Random effects
    u, v, warn2 = _random_effects(p.cip4, p.state)

    # Linear predictor
    lp = intercept + u + v
    contributions: List[Tuple[str, float]] = [("intercept", intercept)]
    if abs(u) > 0:
        contributions.append((f"CIP[{p.cip4}] RE", u))
    if abs(v) > 0:
        contributions.append((f"State[{p.state}] RE", v))

    for fname, val in x.items():
        w = float(coefs.get(fname, 0.0)) * val
        lp += w
        contributions.append((fname, w))

    # Raw & calibrated probability
    p_raw = _sigmoid(lp)
    p_cal = _apply_calibration(p_raw)
    bucket = _bucket(p_cal)

    # Drivers: top 3 (exclude intercept)
    disp = [(n, c) for (n, c) in contributions if n != "intercept"]
    disp.sort(key=lambda t: abs(t[1]), reverse=True)
    top = disp[:3]

    def pretty_factor(name: str) -> str:
        if name.startswith("degree_level_"):
            return "degree_level=" + name.split("_", 1)[1]
        if name.startswith("public_private_"):
            return "public_private=" + name.split("_", 1)[1]
        if name.startswith("size_bin_"):
            return "size_bin=" + name.split("_", 1)[1]
        # keep RE labels as-is
        return name

    drivers = [
        {"factor": pretty_factor(name),
         "direction": "+" if val >= 0 else "-",
         "weight": round(float(val), 4)}
        for name, val in top
    ]

    warnings = warn1 + warn2
    meta = _loaded.get("metadata") or {}

    return {
        "status": "success",
        "model": meta.get("model", "college_earnings"),
        "version": meta.get("version", "v1_75k_5y"),
        "probability": float(p_cal),
        "risk_bucket": bucket,
        "drivers": drivers,
        "confidence": _confidence_from_meta(p.cip4, p.state),
        "warnings": warnings,
    }


def _confidence_from_meta(cip4: str, state: str) -> str:
    """Simple confidence heuristic — refine later if you export SDs or cohort sizes per key.
    Uses presence/absence of random-effects entries as a proxy.
    """
    rc = _loaded.get("rand_cip") or {}
    rs = _loaded.get("rand_state") or {}
    has_cip = cip4 in rc
    has_state = state in rs
    if has_cip and has_state:
        return "High"
    if has_cip or has_state:
        return "Medium"
    return "Low"


