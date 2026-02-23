"""
VSAT Satellite Tracker — Web API

FastAPI app wrapping SatellitePropagator for a web-based satellite map.
Preloads all VSAT TLEs on startup, then serves positions from memory.
"""

import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sat_tracker import (
    VSAT_FLEET,
    SERVICE_PROVIDERS,
    SatellitePropagator,
    compute_slant_range,
    geodetic_to_ecef,
    C_KM_S,
    WGS84_A,
    WGS84_F,
    WGS84_E2,
    WGS84_B,
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
# Geolocation from RTT
# ---------------------------------------------------------------------------

class PingMeasurement(BaseModel):
    norad_id: int
    rtt_ms: float
    ground_delay_ms: float = 0.0
    teleport_lat: float | None = None
    teleport_lon: float | None = None
    time: str | None = None


class GeolocateRequest(BaseModel):
    pings: List[PingMeasurement]


def _slant_range_circle(sat_lat, sat_lon, sat_alt, target_range_km, n_points=360):
    """
    Compute the circle of points on Earth's surface at a given slant range
    from a satellite. Returns list of [lat, lon] pairs.
    """
    sat_ecef = geodetic_to_ecef(sat_lat, sat_lon, sat_alt)
    points = []

    for i in range(n_points):
        azimuth = 2 * math.pi * i / n_points

        # Binary search on angular distance from sub-satellite point
        lo, hi = 0.0, 80.0  # degrees
        for _ in range(50):
            mid = (lo + hi) / 2
            # Great-circle destination from sub-satellite point
            lat1 = math.radians(sat_lat)
            lon1 = math.radians(sat_lon)
            d = math.radians(mid)

            lat2 = math.asin(
                math.sin(lat1) * math.cos(d) +
                math.cos(lat1) * math.sin(d) * math.cos(azimuth)
            )
            lon2 = lon1 + math.atan2(
                math.sin(azimuth) * math.sin(d) * math.cos(lat1),
                math.cos(d) - math.sin(lat1) * math.sin(lat2)
            )

            pt_lat = math.degrees(lat2)
            pt_lon = math.degrees(lon2)
            # Normalize longitude
            pt_lon = ((pt_lon + 180) % 360) - 180

            pt_ecef = geodetic_to_ecef(pt_lat, pt_lon, 0.0)
            dx = sat_ecef - pt_ecef
            rng = float(np.sqrt(np.sum(dx ** 2)))

            if rng < target_range_km:
                lo = mid
            else:
                hi = mid

        # Use the converged point
        lat1 = math.radians(sat_lat)
        lon1 = math.radians(sat_lon)
        d = math.radians((lo + hi) / 2)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(d) +
            math.cos(lat1) * math.sin(d) * math.cos(azimuth)
        )
        lon2 = lon1 + math.atan2(
            math.sin(azimuth) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * math.sin(lat2)
        )
        pt_lat = math.degrees(lat2)
        pt_lon = math.degrees(lon2)
        pt_lon = ((pt_lon + 180) % 360) - 180
        points.append([round(pt_lat, 4), round(pt_lon, 4)])

    # Close the circle
    if points:
        points.append(points[0])
    return points


@app.post("/api/geolocate")
def geolocate(req: GeolocateRequest):
    """
    Given one or more ping measurements (satellite + RTT + ground delay),
    compute slant range circles. Multiple pings produce intersecting circles
    that constrain the ship's location.

    The ping path is: You -> internet -> teleport -> satellite -> ship (and back).
    - ground_delay_ms: round-trip internet/terrestrial portion (you <-> teleport)
    - Satellite leg RTT = total RTT - ground_delay_ms
    - One-way through satellite = sat_leg_rtt / 2 = teleport->satellite + satellite->ship
    - If teleport location given, we subtract teleport->satellite distance
    - What remains = satellite->ship slant range -> draw circle
    """
    circles = []
    for ping in req.pings:
        dt = parse_time(ping.time)
        try:
            pos = propagator.propagate(ping.norad_id, dt)
        except ValueError as e:
            circles.append({"norad_id": ping.norad_id, "error": str(e)})
            continue
        if pos.get("error", 0) != 0:
            circles.append({"norad_id": ping.norad_id, "error": pos.get("error_msg", "SGP4 error")})
            continue

        # Isolate satellite leg
        sat_leg_rtt = ping.rtt_ms - ping.ground_delay_ms
        if sat_leg_rtt <= 0:
            circles.append({"norad_id": ping.norad_id, "error": "Ground delay exceeds total RTT"})
            continue

        # One-way through satellite = teleport->sat + sat->ship
        one_way_total_ms = sat_leg_rtt / 2
        one_way_total_km = one_way_total_ms / 1000 * C_KM_S

        # Subtract teleport-to-satellite distance if teleport location given
        uplink_km = 0.0
        teleport_info = None
        if ping.teleport_lat is not None and ping.teleport_lon is not None:
            rng = compute_slant_range(
                ping.teleport_lat, ping.teleport_lon, 0.0,
                pos["lat_deg"], pos["lon_deg"], pos["alt_km"],
            )
            uplink_km = rng["slant_range_km"]
            teleport_info = {
                "lat": ping.teleport_lat,
                "lon": ping.teleport_lon,
                "uplink_range_km": round(uplink_km, 1),
                "uplink_delay_ms": round(rng["one_way_delay_ms"], 3),
            }

        slant_range_km = one_way_total_km - uplink_km

        # Sanity checks — warn but always draw
        warning = None
        if slant_range_km < pos["alt_km"] * 0.9:
            warning = f"Range {slant_range_km:.0f} km below expected min — clamped to satellite altitude"
            slant_range_km = pos["alt_km"]
        elif slant_range_km > 42500 and pos["orbit_type"] == "GEO":
            if not teleport_info:
                warning = "No teleport — range includes uplink path. Circle wider than true location."
            else:
                warning = "Range exceeds typical max. Ship at very low elevation or delays need tuning."

        # Compute the circle on Earth's surface
        points = _slant_range_circle(
            pos["lat_deg"], pos["lon_deg"], pos["alt_km"],
            slant_range_km, n_points=360,
        )

        result = {
            "norad_id": ping.norad_id,
            "name": pos["name"],
            "time": dt.isoformat(),
            "satellite": {
                "lat": pos["lat_deg"],
                "lon": pos["lon_deg"],
                "alt_km": pos["alt_km"],
            },
            "measured_rtt_ms": ping.rtt_ms,
            "ground_delay_ms": ping.ground_delay_ms,
            "satellite_leg_rtt_ms": round(sat_leg_rtt, 1),
            "one_way_through_sat_ms": round(one_way_total_ms, 3),
            "one_way_through_sat_km": round(one_way_total_km, 1),
            "ship_slant_range_km": round(slant_range_km, 1),
            "ship_one_way_delay_ms": round(slant_range_km / C_KM_S * 1000, 3),
            "circle": points,
        }
        if teleport_info:
            result["teleport"] = teleport_info
        if warning:
            result["warning"] = warning
        circles.append(result)

    return {"circles": circles}


# ---------------------------------------------------------------------------
# Static files — serve index.html at root, everything else from /static
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
