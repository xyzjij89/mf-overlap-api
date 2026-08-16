"""
India Mutual Fund Overlap API + Supabase
Admin bulk upload → Store all schemes → Users search & compare
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import io
import re
from rapidfuzz import process, fuzz
from supabase import create_client, Client
import os
from datetime import datetime

app = FastAPI(
    title="India MF Overlap API (Supabase)",
    description="Full coverage Mutual Fund Overlap tool with Supabase storage",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== SUPABASE ======================
# Set these as environment variables in Railway/Render/Vercel
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # service_role key (admin)

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_db():
    if not supabase:
        raise HTTPException(500, "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
    return supabase


# ====================== MODELS ======================

class FundHoldings(BaseModel):
    name: str
    category: Optional[str] = "Unknown"
    holdings: Dict[str, float]

class OverlapRequest(BaseModel):
    fund_a_name: str
    fund_b_name: str
    as_of_month: Optional[str] = None   # if None → latest
    fuzzy_threshold: int = Field(default=82, ge=70, le=95)

class CommonStock(BaseModel):
    stock: str
    weight_a: float
    weight_b: float
    contribution: float

class OverlapResponse(BaseModel):
    fund_a: str
    fund_b: str
    as_of_month: str
    overlap_pct: float
    interpretation: str
    common_stocks: List[CommonStock]
    unique_a: List[str]
    unique_b: List[str]
    common_count: int


# ====================== PARSING HELPERS ======================

def clean_column_name(col: str) -> str:
    col = str(col).lower().strip()
    col = re.sub(r'[^a-z0-9\s%]', ' ', col)
    return re.sub(r'\s+', ' ', col).strip()

def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    cleaned = {clean_column_name(c): c for c in df.columns}
    for name in possible_names:
        target = clean_column_name(name)
        for cname, original in cleaned.items():
            if target in cname or cname in target:
                return original
    return None

def is_equity(name: str) -> bool:
    name = str(name).lower()
    skip = [
        'treps', 'reverse repo', 'cash', 'net current', 'margin', 'derivative',
        'futures', 'option', 'cblo', 'mutual fund units', 'government security',
        'treasury bill', 'commercial paper', 'certificate of deposit', 'bonds',
        'debenture', 'ncd', 'reits', 'invits', 'others', 'receivables', 'payables',
        'fixed deposit'
    ]
    return not any(kw in name for kw in skip)

def normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    for w in ["ltd", "limited", "ltd.", "inc", "corp", "corporation", "of india", "india", "the"]:
        name = name.replace(w, "")
    return " ".join(name.split())

def interpret(pct: float) -> str:
    if pct < 20: return "Low Overlap — Good diversification"
    elif pct < 35: return "Moderate Overlap — Acceptable for same category"
    elif pct < 50: return "High Overlap — Consider reducing one"
    else: return "Very High Overlap — Significant redundancy"

def guess_category(sheet_name: str) -> str:
    sn = sheet_name.lower()
    if "flexi" in sn: return "Flexi Cap"
    if "large" in sn and "mid" not in sn: return "Large Cap"
    if "mid" in sn and "small" not in sn: return "Mid Cap"
    if "small" in sn: return "Small Cap"
    if "multi" in sn: return "Multi Cap"
    if "elss" in sn or "tax" in sn: return "ELSS"
    if "index" in sn or "nifty" in sn or "sensex" in sn: return "Index"
    if "liquid" in sn or "overnight" in sn: return "Liquid"
    if "debt" in sn or "bond" in sn or "gilt" in sn or "income" in sn: return "Debt"
    if "hybrid" in sn or "balanced" in sn or "aggressive" in sn: return "Hybrid"
    return "Other"

def guess_scheme_type(category: str) -> str:
    if category in ["Flexi Cap", "Large Cap", "Mid Cap", "Small Cap", "Multi Cap", "ELSS", "Index"]:
        return "Equity"
    if category in ["Debt", "Liquid"]:
        return "Debt"
    if category == "Hybrid":
        return "Hybrid"
    return "Other"

def calculate_overlap(holdings_a: Dict[str, float], holdings_b: Dict[str, float], threshold: int = 82):
    stocks_a = list(holdings_a.keys())
    stocks_b = list(holdings_b.keys())
    matches = {}
    remaining_b = set(stocks_b)

    for sa in stocks_a:
        for sb in list(remaining_b):
            if normalize_name(sa) == normalize_name(sb):
                matches[sa] = sb
                remaining_b.remove(sb)
                break

    for sa in stocks_a:
        if sa in matches: continue
        if not remaining_b: break
        best = process.extractOne(sa, list(remaining_b), scorer=fuzz.token_sort_ratio)
        if best and best[1] >= threshold:
            matches[sa] = best[0]
            remaining_b.remove(best[0])

    common = []
    total = 0.0
    for sa, sb in matches.items():
        wa = holdings_a.get(sa, 0)
        wb = holdings_b.get(sb, 0)
        mn = min(wa, wb)
        total += mn
        common.append({
            "stock": sa,
            "weight_a": round(wa, 2),
            "weight_b": round(wb, 2),
            "contribution": round(mn, 2)
        })

    common = sorted(common, key=lambda x: x["contribution"], reverse=True)
    unique_a = [s for s in stocks_a if s not in matches]
    unique_b = list(remaining_b)
    return round(total, 2), common, unique_a, unique_b


def parse_excel_bytes(content: bytes, only_equity: bool = False, min_weight: float = 0.01) -> Dict[str, dict]:
    """only_equity=False → all holdings (Equity + Debt + Hybrid etc.)"""
    results = {}
    try:
        xl = pd.ExcelFile(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid Excel: {str(e)}")

    for sheet_name in xl.sheet_names:
        try:
            df = None
            for header_row in [0, 1, 2, 3]:
                try:
                    temp = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=header_row)
                    if len(temp.columns) >= 4 and not all(str(c).startswith("Unnamed") for c in temp.columns[:3]):
                        df = temp
                        break
                except:
                    continue
            if df is None or df.empty:
                continue

            df.columns = [str(c).strip() for c in df.columns]
            name_col = find_column(df, [
                "name of the instrument", "name of instrument", "instrument name",
                "security name", "scrip name", "company name", "name"
            ])
            weight_col = find_column(df, [
                "% to nav", "% of nav", "% to net assets", "percentage",
                "% of aum", "weight", "%nav", "% to total"
            ])
            if not name_col or not weight_col:
                continue

            clean = pd.DataFrame({"Stock": df[name_col], "Weight": df[weight_col]})
            clean = clean.dropna(subset=["Stock"])
            clean = clean[clean["Stock"].astype(str).str.strip() != ""]
            clean["Weight"] = (
                clean["Weight"].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            clean["Weight"] = pd.to_numeric(clean["Weight"], errors="coerce")
            clean = clean.dropna(subset=["Weight"])
            clean = clean[clean["Weight"] >= min_weight]

            if only_equity:
                clean = clean[clean["Stock"].apply(is_equity)]

            if clean.empty:
                continue

            holdings = dict(zip(
                clean["Stock"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
                clean["Weight"].round(2)
            ))

            cat = guess_category(sheet_name)
            results[sheet_name.strip()] = {
                "category": cat,
                "scheme_type": guess_scheme_type(cat),
                "holdings": holdings
            }
        except Exception:
            continue
    return results


# ====================== ENDPOINTS ======================

@app.get("/")
def root():
    return {
        "message": "India MF Overlap API + Supabase",
        "version": "2.0",
        "docs": "/docs"
    }


@app.post("/admin/bulk-upload")
async def bulk_upload(
    files: List[UploadFile] = File(...),
    as_of_month: str = Form(...),          # e.g. "2026-07"
    amc_name: Optional[str] = Form(None),  # optional, otherwise from filename
    only_equity: bool = Form(False),       # False = all schemes
    min_weight: float = Form(0.01)
):
    """
    ADMIN ONLY: Upload multiple AMC Excel files at once.
    All schemes will be upserted into Supabase.
    """
    db = get_db()
    total_added = 0
    total_updated = 0
    results_summary = []

    for file in files:
        if not file.filename.endswith((".xlsx", ".xls")):
            continue

        content = await file.read()
        parsed = parse_excel_bytes(content, only_equity=only_equity, min_weight=min_weight)

        amc = amc_name or file.filename.replace(".xlsx", "").replace(".xls", "")

        for scheme_name, data in parsed.items():
            record = {
                "scheme_name": scheme_name,
                "amc_name": amc,
                "category": data["category"],
                "scheme_type": data["scheme_type"],
                "holdings": data["holdings"],
                "holdings_count": len(data["holdings"]),
                "as_of_month": as_of_month,
                "source_file": file.filename,
                "updated_at": datetime.utcnow().isoformat()
            }

            # Upsert on (scheme_name + as_of_month)
            existing = db.table("schemes")\
                .select("id")\
                .eq("scheme_name", scheme_name)\
                .eq("as_of_month", as_of_month)\
                .execute()

            if existing.data:
                db.table("schemes").update(record).eq("id", existing.data[0]["id"]).execute()
                total_updated += 1
            else:
                db.table("schemes").insert(record).execute()
                total_added += 1

        results_summary.append({
            "file": file.filename,
            "schemes_parsed": len(parsed)
        })

    # Log
    db.table("upload_logs").insert({
        "filename": ", ".join([f.filename for f in files]),
        "schemes_added": total_added,
        "schemes_updated": total_updated,
        "as_of_month": as_of_month
    }).execute()

    return {
        "message": "Bulk upload complete",
        "as_of_month": as_of_month,
        "schemes_added": total_added,
        "schemes_updated": total_updated,
        "files_processed": results_summary
    }


@app.get("/schemes/search")
def search_schemes(
    q: str = Query(..., min_length=2),
    as_of_month: Optional[str] = None,
    category: Optional[str] = None,
    scheme_type: Optional[str] = None,
    limit: int = Query(20, le=50)
):
    """Search schemes by name. Returns list for dropdown."""
    db = get_db()
    query = db.table("schemes").select("id, scheme_name, amc_name, category, scheme_type, holdings_count, as_of_month")

    query = query.ilike("scheme_name", f"%{q}%")

    if as_of_month:
        query = query.eq("as_of_month", as_of_month)
    if category:
        query = query.eq("category", category)
    if scheme_type:
        query = query.eq("scheme_type", scheme_type)

    # Prefer latest month if not specified
    res = query.order("as_of_month", desc=True).limit(limit).execute()
    return {"count": len(res.data), "schemes": res.data}


@app.get("/schemes/{scheme_id}")
def get_scheme(scheme_id: int):
    """Get full holdings of a scheme"""
    db = get_db()
    res = db.table("schemes").select("*").eq("id", scheme_id).single().execute()
    if not res.data:
        raise HTTPException(404, "Scheme not found")
    return res.data


@app.post("/overlap", response_model=OverlapResponse)
def overlap(req: OverlapRequest):
    """
    Compare two schemes by name (from database).
    Uses latest available month if as_of_month not given.
    """
    db = get_db()

    def fetch_scheme(name: str, month: Optional[str]):
        q = db.table("schemes").select("*").ilike("scheme_name", f"%{name}%")
        if month:
            q = q.eq("as_of_month", month)
        res = q.order("as_of_month", desc=True).limit(1).execute()
        if not res.data:
            raise HTTPException(404, f"Scheme not found: {name}")
        return res.data[0]

    sa = fetch_scheme(req.fund_a_name, req.as_of_month)
    sb = fetch_scheme(req.fund_b_name, req.as_of_month)

    # Prefer same month
    month = sa["as_of_month"]
    if sa["as_of_month"] != sb["as_of_month"]:
        # try to align
        sb2 = fetch_scheme(req.fund_b_name, sa["as_of_month"])
        if sb2:
            sb = sb2

    ov, common, uniq_a, uniq_b = calculate_overlap(
        sa["holdings"], sb["holdings"], threshold=req.fuzzy_threshold
    )

    return OverlapResponse(
        fund_a=sa["scheme_name"],
        fund_b=sb["scheme_name"],
        as_of_month=month,
        overlap_pct=ov,
        interpretation=interpret(ov),
        common_stocks=[CommonStock(**c) for c in common],
        unique_a=uniq_a[:40],
        unique_b=uniq_b[:40],
        common_count=len(common)
    )


@app.get("/stats")
def stats():
    """Quick stats for admin dashboard"""
    db = get_db()
    total = db.table("schemes").select("id", count="exact").execute()
    months = db.table("schemes").select("as_of_month").execute()
    unique_months = sorted(list(set([r["as_of_month"] for r in months.data if r["as_of_month"]])), reverse=True)
    return {
        "total_schemes": total.count,
        "available_months": unique_months[:6]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_supabase:app", host="0.0.0.0", port=8000, reload=True)
