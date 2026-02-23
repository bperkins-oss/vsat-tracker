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

    Uses adaptive tolerance based on actual range spread, and weights
    candidates by total error to tighten the confidence region.
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

    # Adaptive tolerance: based on range spread (RTT jitter)
    range_spread = max(ranges) - min(ranges) if len(ranges) > 1 else 500
    range_tolerance_km = max(200, range_spread * 1.5)

    # Sample points along each circle and score by proximity to all circles
    candidates = []
    for i, pts_i in enumerate(all_circle_points):
        step = max(1, len(pts_i) // 90)  # ~90 samples per circle
        for pt in pts_i[::step]:
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

            if score >= min(len(circles_data), 2):
                candidates.append((lat, lon, score, total_err))

    if not candidates:
        # Widen tolerance and try again
        range_tolerance_km *= 2
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

    # Take candidates that score well AND have low total error
    n_circles = len(circles_data)
    high_score = [c for c in candidates if c[2] >= max(2, n_circles - 1)]
    if not high_score:
        high_score = candidates[:50]

    # Further filter to the best 50% by error (tightest cluster)
    high_score.sort(key=lambda c: c[3])
    best = high_score[:max(10, len(high_score) // 2)]

    lats = [c[0] for c in best]
    lons = [c[1] for c in best]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Compute radius — use 90th percentile instead of max for robustness
    dists = []
    for lat, lon, _, _ in best:
        dlat = (lat - center_lat) * 111.0
        dlon = (lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        dists.append(math.sqrt(dlat ** 2 + dlon ** 2))
    dists.sort()
    p90_idx = int(len(dists) * 0.9)
    radius_km = dists[p90_idx] if p90_idx < len(dists) else dists[-1]

    # Generate confidence circle polygon
    confidence_points = []
    for i in range(72):
        angle = 2 * math.pi * i / 72
        r_lat = radius_km / 111.0
        r_lon = radius_km / (111.0 * max(0.01, math.cos(math.radians(center_lat))))
        pt_lat = center_lat + r_lat * math.sin(angle)
        pt_lon = center_lon + r_lon * math.cos(angle)
        confidence_points.append([round(pt_lat, 4), round(pt_lon, 4)])
    confidence_points.append(confidence_points[0])

    return {
        "center_lat": round(center_lat, 4),
        "center_lon": round(center_lon, 4),
        "radius_km": round(radius_km, 1),
        "radius_nm": round(radius_km / 1.852, 1),
        "radius_mi": round(radius_km / 1.609, 1),
        "n_candidates": len(best),
        "best_score": best[0][2] if best else 0,
        "range_tolerance_km": round(range_tolerance_km, 1),
        "polygon": confidence_points,
    }


def _tracking_worker(track_id: str, ip: str, interval_sec: int,
                     n_rounds: int, pings_per_round: int,
                     ground_delay_ms: float, teleport_lat: float,
                     teleport_lon: float, auto_calibrate: bool = False):
    """Background thread that pings a ship periodically and updates tracking session."""
    session = tracking_sessions[track_id]

    # Auto-calibrate if requested
    if auto_calibrate:
        session["status"] = "Calibrating — measuring ground delay and testing gateways..."
        ground_data = _measure_ground_delay(ip, 5)
        cal_ground = ground_data.get("estimated_ground_delay_ms", ground_delay_ms)

        # Quick ping for RTT
        quick_ping = _ping_host(ip, 10)
        if "error" not in quick_ping:
            cal_rtt = quick_ping["min_ms"]
            sat_leg = cal_rtt - cal_ground
            if sat_leg > 0:
                ow_km = (sat_leg / 2) / 1000 * C_KM_S
                now = datetime.now(timezone.utc)
                o3b_sats = _find_best_o3b(now)

                # Test all gateways, find valid ones
                best_gw = None
                best_ship_range = float("inf")
                for gw_key, gw in O3B_GATEWAYS.items():
                    best_elev = -90
                    best_sat_for_gw = None
                    for sat in o3b_sats:
                        rng = compute_slant_range(
                            gw["lat"], gw["lon"], 0.0,
                            sat["lat_deg"], sat["lon_deg"], sat["alt_km"],
                        )
                        if rng["elevation_deg"] > best_elev:
                            best_elev = rng["elevation_deg"]
                            best_sat_for_gw = (sat, rng)
                    if best_sat_for_gw and best_elev >= 5:
                        sat, rng = best_sat_for_gw
                        ship_range = ow_km - rng["slant_range_km"]
                        if ship_range >= sat["alt_km"] and ship_range < best_ship_range:
                            best_ship_range = ship_range
                            best_gw = (gw_key, gw, sat)

                if best_gw:
                    gw_key, gw, sat = best_gw
                    teleport_lat = gw["lat"]
                    teleport_lon = gw["lon"]
                    ground_delay_ms = cal_ground
                    session["calibration"] = {
                        "gateway": gw_key,
                        "gateway_label": gw["label"],
                        "ground_delay_ms": round(cal_ground, 2),
                        "ground_data": ground_data,
                    }
                    session["teleport"] = {"lat": teleport_lat, "lon": teleport_lon}
                    session["ground_delay_ms"] = ground_delay_ms
                else:
                    # No valid gateway found — use measured ground delay with original gateway
                    ground_delay_ms = cal_ground
                    session["calibration"] = {
                        "gateway": "original",
                        "ground_delay_ms": round(cal_ground, 2),
                        "note": "No unclamped gateway found — using calibrated ground delay only",
                        "ground_data": ground_data,
                    }
                    session["ground_delay_ms"] = ground_delay_ms

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
        # Sort by elevation descending (highest elevation = shortest uplink = most
        # range budget for ship). Pick the first one that gives an unclamped circle,
        # or the one with shortest uplink if all are clamped.
        candidates_by_elev = []
        for sat in o3b_sats:
            rng = compute_slant_range(
                teleport_lat, teleport_lon, 0.0,
                sat["lat_deg"], sat["lon_deg"], sat["alt_km"],
            )
            if rng["elevation_deg"] < 5.0:
                continue
            candidates_by_elev.append((rng, sat))
        candidates_by_elev.sort(key=lambda x: -x[0]["elevation_deg"])

        best_circle = None
        best_unclamped = None
        best_clamped = None
        shortest_uplink = float("inf")

        for rng, sat in candidates_by_elev:
            uplink_km = rng["slant_range_km"]
            ship_range = one_way_total_km - uplink_km

            min_range = sat["alt_km"] * 1.005
            clamped = False
            if ship_range < min_range:
                if ship_range > 0:
                    ship_range = min_range
                    clamped = True
                else:
                    continue

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
                # Track best clamped by shortest uplink
                if uplink_km < shortest_uplink:
                    shortest_uplink = uplink_km
                    best_clamped = candidate
            else:
                # Prefer unclamped with shortest uplink (highest gateway elevation)
                if best_unclamped is None:
                    best_unclamped = candidate

        best_circle = best_unclamped or best_clamped

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


# Known O3b mPOWER gateway locations (SES announced sites)
O3B_GATEWAYS = {
    "thermopylae_greece": {"lat": 38.90, "lon": 22.56, "label": "Thermopylae, Greece"},
    "hawaii_us": {"lat": 21.52, "lon": -158.00, "label": "Hawaii, US (SES)"},
    "dubbo_australia": {"lat": -32.24, "lon": 148.60, "label": "Dubbo, Australia"},
    "merredin_australia": {"lat": -31.48, "lon": 118.28, "label": "Merredin, Australia"},
    "phoenix_us": {"lat": 33.45, "lon": -112.07, "label": "Phoenix, Arizona, US"},
    "gandoul_senegal": {"lat": 14.58, "lon": -17.10, "label": "Gandoul, Senegal"},
    "chile": {"lat": -33.45, "lon": -70.67, "label": "Santiago, Chile"},
    "uae": {"lat": 24.45, "lon": 54.65, "label": "Abu Dhabi, UAE"},
    "south_africa": {"lat": -25.75, "lon": 28.23, "label": "Johannesburg, South Africa"},
    "peru": {"lat": -12.05, "lon": -77.03, "label": "Lima, Peru"},
    "brazil": {"lat": -23.55, "lon": -46.63, "label": "São Paulo, Brazil"},
    "portugal": {"lat": 38.75, "lon": -9.14, "label": "Lisbon, Portugal"},
}


@app.get("/api/o3b_gateways")
def o3b_gateways():
    """Return known O3b mPOWER gateway locations."""
    return O3B_GATEWAYS


def _measure_ground_delay(ip: str, count: int = 10) -> dict:
    """
    Measure ground delay by pinging with increasing TTL to find the last
    ground hop before the satellite jump. Returns estimated ground RTT.
    """
    # First get the baseline RTT to the target
    base_ping = _ping_host(ip, count)
    if "error" in base_ping:
        return {"error": base_ping["error"]}

    base_rtt = base_ping["min_ms"]

    # Probe with increasing TTL to find the last ground hop
    last_ground_rtt = 0
    last_ground_hop = None
    hops = []

    for ttl in range(1, 25):
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", "-t", str(ttl), ip],
                capture_output=True, text=True, timeout=5,
            )
            # Check if we reached the target
            rtt_match = re.search(r"time=([0-9.]+)", result.stdout)
            exceeded = re.search(r"From ([0-9.]+)", result.stdout)

            if rtt_match:
                # Reached destination
                hops.append({
                    "ttl": ttl, "ip": ip, "rtt_ms": float(rtt_match.group(1)),
                    "type": "destination"
                })
                break
            elif exceeded:
                hop_ip = exceeded.group(1)
                # Ping this hop directly to get its RTT
                hop_result = subprocess.run(
                    ["ping", "-c", "3", "-W", "2", hop_ip],
                    capture_output=True, text=True, timeout=10,
                )
                hop_rtt_match = re.search(
                    r"min/avg/max.*?= ([0-9.]+)", hop_result.stdout
                )
                hop_rtt = float(hop_rtt_match.group(1)) if hop_rtt_match else None

                hops.append({
                    "ttl": ttl, "ip": hop_ip, "rtt_ms": hop_rtt,
                    "type": "hop"
                })

                # This is a ground hop if RTT is much less than target RTT
                if hop_rtt and hop_rtt < base_rtt * 0.5:
                    last_ground_rtt = hop_rtt
                    last_ground_hop = hop_ip
            else:
                hops.append({"ttl": ttl, "ip": "*", "rtt_ms": None, "type": "timeout"})
        except Exception:
            hops.append({"ttl": ttl, "ip": "*", "rtt_ms": None, "type": "error"})

    # The ground delay is the RTT to the last hop before the satellite jump
    # Cap it at (base_rtt - min_theoretical_satellite_rtt)
    # For O3b at ~8062km altitude: min sat RTT = 2 * 2 * 8062 / 299792 * 1000 ≈ 107.6ms
    min_sat_rtt_ms = 107.6
    max_ground_delay = max(0, base_rtt - min_sat_rtt_ms)

    estimated_ground_delay = min(last_ground_rtt, max_ground_delay) if last_ground_rtt else max_ground_delay

    return {
        "base_rtt_ms": round(base_rtt, 2),
        "last_ground_hop": last_ground_hop,
        "last_ground_hop_rtt_ms": round(last_ground_rtt, 2) if last_ground_rtt else None,
        "max_ground_delay_ms": round(max_ground_delay, 2),
        "estimated_ground_delay_ms": round(estimated_ground_delay, 2),
        "hops": hops,
    }


@app.get("/api/calibrate")
def calibrate(ip: str = Query(...), pings: int = Query(20)):
    """
    Auto-calibrate gateway + ground delay for a ship IP.
    Pings the ship, measures ground delay via TTL probing, then tests
    all known O3b gateways to find which one produces valid (unclamped)
    ship ranges. Returns ranked results.
    """
    now = datetime.now(timezone.utc)

    # Step 1: Ping for baseline RTT
    ping_result = _ping_host(ip, pings)
    if "error" in ping_result:
        return JSONResponse(status_code=400, content={"error": f"Cannot ping {ip}: {ping_result['error']}"})

    min_rtt = ping_result["min_ms"]

    # Step 2: Measure ground delay
    ground_data = _measure_ground_delay(ip, 5)
    ground_delay = ground_data.get("estimated_ground_delay_ms", 5.0)

    # Step 3: Find all O3b satellites at current time
    o3b_sats = _find_best_o3b(now)

    # Step 4: Test each gateway
    sat_leg_rtt = min_rtt - ground_delay
    if sat_leg_rtt <= 0:
        # Ground delay too high — fall back to theoretical max
        ground_delay = ground_data.get("max_ground_delay_ms", 5.0)
        sat_leg_rtt = min_rtt - ground_delay

    one_way_km = (sat_leg_rtt / 2) / 1000 * C_KM_S

    results = []
    for gw_key, gw in O3B_GATEWAYS.items():
        # Find best satellite for this gateway (highest elevation)
        best_sat = None
        best_elev = -90
        for sat in o3b_sats:
            rng = compute_slant_range(
                gw["lat"], gw["lon"], 0.0,
                sat["lat_deg"], sat["lon_deg"], sat["alt_km"],
            )
            if rng["elevation_deg"] > best_elev:
                best_elev = rng["elevation_deg"]
                best_sat = (sat, rng)

        if not best_sat or best_elev < 5:
            continue

        sat, rng = best_sat
        uplink_km = rng["slant_range_km"]
        ship_range = one_way_km - uplink_km
        clamped = ship_range < sat["alt_km"]

        results.append({
            "gateway": gw_key,
            "gateway_label": gw["label"],
            "gateway_lat": gw["lat"],
            "gateway_lon": gw["lon"],
            "satellite": sat["name"],
            "satellite_norad": sat["norad_id"],
            "sat_lat": round(sat["lat_deg"], 3),
            "sat_lon": round(sat["lon_deg"], 3),
            "sat_alt_km": round(sat["alt_km"], 1),
            "elevation_deg": round(best_elev, 1),
            "uplink_km": round(uplink_km, 1),
            "ship_range_km": round(max(ship_range, sat["alt_km"]), 1),
            "clamped": clamped,
            "valid": not clamped,
        })

    # Sort: valid (unclamped) first, then by smallest ship range (closest to nadir)
    results.sort(key=lambda r: (not r["valid"], r["ship_range_km"]))

    return {
        "ip": ip,
        "ping": ping_result,
        "ground_delay": ground_data,
        "satellite_leg_rtt_ms": round(sat_leg_rtt, 2),
        "one_way_through_sat_km": round(one_way_km, 1),
        "gateways_tested": len(results),
        "valid_gateways": len([r for r in results if r["valid"]]),
        "results": results,
        "recommended": results[0] if results else None,
    }


class TrackRequest(BaseModel):
    ip: str
    interval_sec: int = 1800        # 30 minutes default
    n_rounds: int = 6               # 6 rounds = 3 hours at 30min intervals
    pings_per_round: int = 50       # 50 pings per round for stable min RTT
    ground_delay_ms: float = 5.0    # overridden by calibration
    teleport_lat: float = 38.90     # Thermopylae, Greece (O3b gateway)
    teleport_lon: float = 22.56
    auto_calibrate: bool = False    # auto-detect gateway + ground delay


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
              req.teleport_lat, req.teleport_lon, req.auto_calibrate),
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
        "calibration": session.get("calibration"),
        "ground_delay_ms": session.get("ground_delay_ms"),
        "teleport": session.get("teleport"),
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
