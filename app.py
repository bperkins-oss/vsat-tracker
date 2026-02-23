"""
VSAT Satellite Tracker — Web API

FastAPI app wrapping SatellitePropagator for a web-based satellite map.
Preloads all VSAT TLEs on startup, then serves positions from memory.
"""

import asyncio
import math
import os
import subprocess
import re
import threading
import time as time_mod
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List

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
# Ship tracking — time-series ping with confidence region
# ---------------------------------------------------------------------------

# Active tracking sessions: track_id -> session data
tracking_sessions: Dict[str, dict] = {}


def _ping_host(ip: str, count: int = 20) -> dict:
    """Ping an IP and return min/avg/max RTT."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), "-W", "4", ip],
            capture_output=True, text=True, timeout=count * 5 + 10,
        )
        rtts = [float(m) for m in re.findall(r"time=([0-9.]+)", result.stdout)]
        if not rtts:
            return {"error": "no response", "ip": ip}
        return {
            "ip": ip,
            "count": len(rtts),
            "min_ms": round(min(rtts), 2),
            "avg_ms": round(sum(rtts) / len(rtts), 2),
            "max_ms": round(max(rtts), 2),
            "all_ms": [round(r, 2) for r in rtts],
        }
    except Exception as e:
        return {"error": str(e), "ip": ip}


def _find_best_o3b(dt: datetime) -> list:
    """Find all O3b satellites visible at a given time, sorted by longitude."""
    sats = []
    for nid, (sat, entry, prov) in propagator._satellites.items():
        if prov != "o3b":
            continue
        pos = propagator.propagate(nid, dt)
        if pos.get("error", 0) == 0:
            sats.append(pos)
    sats.sort(key=lambda s: s["lon_deg"])
    return sats


def _compute_confidence_region(circles_data: list) -> dict | None:
    """
    Given multiple slant range circles, find the intersection region
    by sampling points and scoring how many circles each point lies near.
    Returns a confidence polygon and center estimate.
    """
    if len(circles_data) < 2:
        return None

    # Collect all circle polygons as numpy arrays
    all_circle_points = []
    sat_positions = []
    ranges = []
    for c in circles_data:
        pts = np.array(c["circle"][:-1])  # exclude closing duplicate
        all_circle_points.append(pts)
        sat_positions.append(geodetic_to_ecef(
            c["satellite"]["lat"], c["satellite"]["lon"], c["satellite"]["alt_km"]
        ))
        ranges.append(c["ship_slant_range_km"])

    # Build candidate grid from circle intersections
    # Sample points along each circle and check distance to other circles
    candidates = []
    range_tolerance_km = 500  # ±500 km tolerance band

    for i, pts_i in enumerate(all_circle_points):
        for pt in pts_i[::3]:  # every 3rd point for speed
            lat, lon = pt[0], pt[1]
            pt_ecef = geodetic_to_ecef(lat, lon, 0.0)

            # Score: how many other circles is this point near?
            score = 0
            total_err = 0
            for j, sat_ecef_j in enumerate(sat_positions):
                dx = sat_ecef_j - pt_ecef
                actual_range = float(np.sqrt(np.sum(dx ** 2)))
                err = abs(actual_range - ranges[j])
                if err < range_tolerance_km:
                    score += 1
                total_err += err

            if score >= min(len(circles_data), 2):
                candidates.append((lat, lon, score, total_err))

    if not candidates:
        # Widen tolerance and try again
        range_tolerance_km = 1000
        for i, pts_i in enumerate(all_circle_points):
            for pt in pts_i:
                lat, lon = pt[0], pt[1]
                pt_ecef = geodetic_to_ecef(lat, lon, 0.0)
                score = 0
                total_err = 0
                for j, sat_ecef_j in enumerate(sat_positions):
                    dx = sat_ecef_j - pt_ecef
                    actual_range = float(np.sqrt(np.sum(dx ** 2)))
                    err = abs(actual_range - ranges[j])
                    if err < range_tolerance_km:
                        score += 1
                    total_err += err
                if score >= 2:
                    candidates.append((lat, lon, score, total_err))

    if not candidates:
        return None

    # Sort by score (desc) then total error (asc)
    candidates.sort(key=lambda c: (-c[2], c[3]))

    # Take top candidates and compute centroid + bounding region
    n_circles = len(circles_data)
    best = [c for c in candidates if c[2] >= max(2, n_circles - 1)]
    if not best:
        best = candidates[:50]

    lats = [c[0] for c in best]
    lons = [c[1] for c in best]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Compute radius of confidence region
    max_dist = 0
    for lat, lon, _, _ in best:
        dlat = (lat - center_lat) * 111.0  # km per degree
        dlon = (lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        max_dist = max(max_dist, dist)

    # Generate confidence circle polygon
    confidence_points = []
    for i in range(72):
        angle = 2 * math.pi * i / 72
        # radius in degrees
        r_lat = max_dist / 111.0
        r_lon = max_dist / (111.0 * max(0.01, math.cos(math.radians(center_lat))))
        pt_lat = center_lat + r_lat * math.sin(angle)
        pt_lon = center_lon + r_lon * math.cos(angle)
        confidence_points.append([round(pt_lat, 4), round(pt_lon, 4)])
    confidence_points.append(confidence_points[0])

    return {
        "center_lat": round(center_lat, 4),
        "center_lon": round(center_lon, 4),
        "radius_km": round(max_dist, 1),
        "radius_nm": round(max_dist / 1.852, 1),
        "radius_mi": round(max_dist / 1.609, 1),
        "n_candidates": len(best),
        "best_score": best[0][2] if best else 0,
        "polygon": confidence_points,
    }


def _tracking_worker(track_id: str, ip: str, interval_sec: int,
                     n_rounds: int, pings_per_round: int,
                     ground_delay_ms: float, teleport_lat: float,
                     teleport_lon: float):
    """Background thread that pings a ship periodically and updates tracking session."""
    session = tracking_sessions[track_id]

    for round_num in range(n_rounds):
        if session.get("stopped"):
            break

        now = datetime.now(timezone.utc)
        session["status"] = f"Round {round_num + 1}/{n_rounds} — pinging {ip}..."

        # Ping the ship
        ping_result = _ping_host(ip, pings_per_round)
        if "error" in ping_result:
            session["measurements"].append({
                "round": round_num + 1,
                "time": now.isoformat(),
                "error": ping_result["error"],
            })
            if round_num < n_rounds - 1:
                session["status"] = f"Round {round_num + 1} failed — waiting for next..."
                time_mod.sleep(interval_sec)
            continue

        min_rtt = ping_result["min_ms"]

        # Find all O3b satellites at this time
        o3b_sats = _find_best_o3b(now)

        # For each O3b satellite, compute what the slant range would be
        sat_leg_rtt = min_rtt - ground_delay_ms
        if sat_leg_rtt <= 0:
            session["measurements"].append({
                "round": round_num + 1,
                "time": now.isoformat(),
                "error": "Ground delay exceeds RTT",
            })
            continue

        one_way_total_km = (sat_leg_rtt / 2) / 1000 * C_KM_S

        # Try each O3b satellite — only those visible from teleport (elevation > 5°)
        best_circle = None
        best_range_diff = float("inf")

        for sat in o3b_sats:
            rng = compute_slant_range(
                teleport_lat, teleport_lon, 0.0,
                sat["lat_deg"], sat["lon_deg"], sat["alt_km"],
            )
            # Skip satellites not visible from teleport
            if rng["elevation_deg"] < 5.0:
                continue

            uplink_km = rng["slant_range_km"]
            ship_range = one_way_total_km - uplink_km

            # Minimum circle range = altitude * 1.005 to ensure non-degenerate circle
            min_range = sat["alt_km"] * 1.005
            clamped = False
            if ship_range < min_range:
                if ship_range > 0:
                    ship_range = min_range
                    clamped = True
                else:
                    continue  # impossible geometry

            circle_pts = _slant_range_circle(
                sat["lat_deg"], sat["lon_deg"], sat["alt_km"],
                ship_range, n_points=180,
            )
            candidate = {
                "norad_id": sat["norad_id"],
                "name": sat["name"],
                "satellite": {"lat": sat["lat_deg"], "lon": sat["lon_deg"], "alt_km": sat["alt_km"]},
                "ship_slant_range_km": round(ship_range, 1),
                "uplink_km": round(uplink_km, 1),
                "teleport_elevation_deg": round(rng["elevation_deg"], 1),
                "circle": circle_pts,
            }
            if clamped:
                candidate["clamped"] = True
            # Prefer the satellite that gives a range closest to the altitude
            # (most likely directly above or near the ship)
            diff = abs(ship_range - sat["alt_km"])
            if diff < best_range_diff:
                best_range_diff = diff
                best_circle = candidate

        measurement = {
            "round": round_num + 1,
            "time": now.isoformat(),
            "ping": ping_result,
            "min_rtt_ms": min_rtt,
            "sat_leg_rtt_ms": round(sat_leg_rtt, 1),
            "one_way_km": round(one_way_total_km, 1),
        }
        if best_circle:
            measurement["circle"] = best_circle

        session["measurements"].append(measurement)

        # Recompute confidence region from all circles so far
        all_circles = [m["circle"] for m in session["measurements"] if "circle" in m]
        if len(all_circles) >= 2:
            conf = _compute_confidence_region(all_circles)
            if conf:
                session["confidence"] = conf

        session["status"] = f"Round {round_num + 1}/{n_rounds} complete — {len(all_circles)} circles"

        # Wait for next round (unless last)
        if round_num < n_rounds - 1 and not session.get("stopped"):
            for _ in range(interval_sec):
                if session.get("stopped"):
                    break
                time_mod.sleep(1)

    session["status"] = "complete"
    session["completed_at"] = datetime.now(timezone.utc).isoformat()


class TrackRequest(BaseModel):
    ip: str
    interval_sec: int = 1800        # 30 minutes default
    n_rounds: int = 6               # 6 rounds = 3 hours at 30min intervals
    pings_per_round: int = 20       # 20 pings per round for stable min RTT
    ground_delay_ms: float = 5.0    # minimal ground delay
    teleport_lat: float = 49.68     # SES Betzdorf Luxembourg
    teleport_lon: float = 6.33


@app.post("/api/track")
def start_tracking(req: TrackRequest):
    """Start a background tracking session for a ship IP."""
    track_id = f"track_{req.ip.replace('.', '_')}_{int(time_mod.time())}"
    session = {
        "track_id": track_id,
        "ip": req.ip,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "interval_sec": req.interval_sec,
        "n_rounds": req.n_rounds,
        "pings_per_round": req.pings_per_round,
        "ground_delay_ms": req.ground_delay_ms,
        "teleport": {"lat": req.teleport_lat, "lon": req.teleport_lon},
        "status": "starting",
        "measurements": [],
        "confidence": None,
        "stopped": False,
    }
    tracking_sessions[track_id] = session

    thread = threading.Thread(
        target=_tracking_worker,
        args=(track_id, req.ip, req.interval_sec, req.n_rounds,
              req.pings_per_round, req.ground_delay_ms,
              req.teleport_lat, req.teleport_lon),
        daemon=True,
    )
    thread.start()

    return {"track_id": track_id, "status": "started"}


@app.get("/api/track/{track_id}")
def get_tracking(track_id: str):
    """Get current status and results of a tracking session."""
    session = tracking_sessions.get(track_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Unknown track ID"})

    # Build response without the full circle point arrays (too large for status)
    measurements_summary = []
    circles_full = []
    for m in session["measurements"]:
        summary = {
            "round": m["round"],
            "time": m["time"],
        }
        if "error" in m:
            summary["error"] = m["error"]
        else:
            summary["min_rtt_ms"] = m["min_rtt_ms"]
            summary["sat_leg_rtt_ms"] = m["sat_leg_rtt_ms"]
            if "circle" in m:
                summary["satellite"] = m["circle"]["name"]
                summary["ship_range_km"] = m["circle"]["ship_slant_range_km"]
                circles_full.append(m["circle"])
        measurements_summary.append(summary)

    return {
        "track_id": session["track_id"],
        "ip": session["ip"],
        "started_at": session["started_at"],
        "status": session["status"],
        "n_rounds": session["n_rounds"],
        "interval_sec": session["interval_sec"],
        "completed_rounds": len(session["measurements"]),
        "measurements": measurements_summary,
        "circles": circles_full,
        "confidence": session.get("confidence"),
    }


@app.post("/api/track/{track_id}/stop")
def stop_tracking(track_id: str):
    """Stop a running tracking session."""
    session = tracking_sessions.get(track_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Unknown track ID"})
    session["stopped"] = True
    return {"status": "stopping"}


@app.get("/api/tracks")
def list_tracks():
    """List all tracking sessions."""
    return [{
        "track_id": s["track_id"],
        "ip": s["ip"],
        "status": s["status"],
        "completed_rounds": len(s["measurements"]),
        "n_rounds": s["n_rounds"],
        "confidence": s.get("confidence"),
    } for s in tracking_sessions.values()]


# ---------------------------------------------------------------------------
# Static files — serve index.html at root, everything else from /static
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
