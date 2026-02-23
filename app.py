"""
VSAT Satellite Tracker — Web API

FastAPI app wrapping SatellitePropagator for a web-based satellite map.
Preloads all VSAT TLEs on startup, then serves positions from memory.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sat_tracker import (
    VSAT_FLEET,
    SERVICE_PROVIDERS,
    SatellitePropagator,
    compute_slant_range,
)

# ---------------------------------------------------------------------------
# Global propagator — loaded once at startup
# ---------------------------------------------------------------------------

propagator: SatellitePropagator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global propagator
    print("Loading VSAT fleet TLEs from Celestrak...")
    propagator = SatellitePropagator()
    total = propagator.load_fleet()
    print(f"Loaded {total} satellites — ready to serve")
    yield
    print("Shutting down")


app = FastAPI(title="VSAT Satellite Tracker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time(t: str | None) -> datetime:
    if not t:
        return datetime.now(timezone.utc)
    s = t.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    count = len(propagator._satellites) if propagator else 0
    return {"status": "ok", "satellites_loaded": count}


@app.get("/api/providers")
def providers():
    operators = []
    for name, info in VSAT_FLEET.items():
        operators.append({
            "name": name,
            "desc": info["desc"],
            "orbit": info["orbit"],
            "type": "operator",
        })
    services = []
    for name, info in SERVICE_PROVIDERS.items():
        services.append({
            "name": name,
            "desc": info["desc"],
            "operators": info["operators"],
            "type": "service_provider",
        })
    return {"operators": operators, "service_providers": services}


@app.get("/api/fleet")
def fleet(time: str | None = Query(None)):
    dt = parse_time(time)
    results = []
    for nid in propagator._satellites:
        r = propagator.propagate(nid, dt)
        if r.get("error", 0) == 0:
            results.append(r)
    results.sort(key=lambda r: (r["provider"], r["lon_deg"]))
    return {"time": dt.isoformat(), "count": len(results), "satellites": results}


@app.get("/api/provider/{name}")
def provider(name: str, time: str | None = Query(None)):
    dt = parse_time(time)
    pname = name.lower()

    # Service provider — aggregate from operators
    if pname in SERVICE_PROVIDERS:
        sp = SERVICE_PROVIDERS[pname]
        results = []
        for op_name in sp["operators"]:
            for nid, (sat, entry, prov) in propagator._satellites.items():
                if prov == op_name:
                    r = propagator.propagate(nid, dt)
                    if r.get("error", 0) == 0:
                        results.append(r)
        results.sort(key=lambda r: (r["provider"], r["lon_deg"]))
        return {
            "provider": pname,
            "type": "service_provider",
            "desc": sp["desc"],
            "operators": sp["operators"],
            "time": dt.isoformat(),
            "count": len(results),
            "satellites": results,
        }

    # Direct operator
    if pname not in VSAT_FLEET:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown provider: {pname}"},
        )

    info = VSAT_FLEET[pname]
    results = []
    for nid, (sat, entry, prov) in propagator._satellites.items():
        if prov == pname:
            r = propagator.propagate(nid, dt)
            if r.get("error", 0) == 0:
                results.append(r)
    results.sort(key=lambda r: r["lon_deg"])
    return {
        "provider": pname,
        "type": "operator",
        "desc": info["desc"],
        "orbit": info["orbit"],
        "time": dt.isoformat(),
        "count": len(results),
        "satellites": results,
    }


@app.get("/api/satellite/{norad_id}")
def satellite(norad_id: int, time: str | None = Query(None)):
    dt = parse_time(time)
    try:
        r = propagator.propagate(norad_id, dt)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    if r.get("error", 0) != 0:
        return JSONResponse(status_code=500, content=r)
    return r


@app.get("/api/range")
def slant_range(
    norad_id: int = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    alt: float = Query(0.0),
    time: str | None = Query(None),
):
    dt = parse_time(time)
    try:
        pos = propagator.propagate(norad_id, dt)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    if pos.get("error", 0) != 0:
        return JSONResponse(status_code=500, content=pos)

    rng = compute_slant_range(lat, lon, alt,
                               pos["lat_deg"], pos["lon_deg"], pos["alt_km"])
    return {
        "satellite": pos,
        "observer": {"lat": lat, "lon": lon, "alt_km": alt},
        "range": rng,
    }


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    if not propagator:
        return {"results": []}
    q_upper = q.upper()
    matches = []
    for nid, (sat, entry, prov) in propagator._satellites.items():
        name = entry.get("OBJECT_NAME", "")
        if q_upper in name.upper() or q == str(nid):
            matches.append({
                "norad_id": nid,
                "name": name,
                "provider": prov,
            })
    matches.sort(key=lambda m: m["name"])
    return {"query": q, "count": len(matches), "results": matches[:50]}


# ---------------------------------------------------------------------------
# Static files — serve index.html at root, everything else from /static
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
