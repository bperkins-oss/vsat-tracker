#!/usr/bin/env python3
"""
VSAT Satellite Position Tracker — exact positions for maritime VSAT satellites.

Setup:
    source ~/sat-tracker-venv/bin/activate

Usage:
    python sat_tracker.py fleet                           # all VSAT sats, current positions
    python sat_tracker.py provider inmarsat               # one provider's fleet
    python sat_tracker.py position --name "INMARSAT 4-F2" # single sat
    python sat_tracker.py position --norad 28899          # by NORAD ID
    python sat_tracker.py position --norad 28899 --time 2026-02-20T14:30:00Z
    python sat_tracker.py range --norad 28899 --observer 25,-80,0
    python sat_tracker.py track --norad 28899 --start 2026-02-22T00:00Z --end 2026-02-22T06:00Z
    python sat_tracker.py search "INMARSAT"
    python sat_tracker.py refresh
    python sat_tracker.py status
    python sat_tracker.py test
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path

import numpy as np
import requests
from sgp4 import omm
from sgp4.api import Satrec, jday

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_KM_S = 299_792.458
WGS84_A = 6_378.137
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"

# ---------------------------------------------------------------------------
# VSAT Fleet — provider-to-satellite mapping
#
# Each provider lists the satellites it operates or leases capacity on.
# NORAD IDs verified against Celestrak, February 2026.
# ---------------------------------------------------------------------------

VSAT_FLEET = {
    # --- Primary maritime satellite operators ---
    "inmarsat": {
        "desc": "Inmarsat (Viasat) — L-band + Ka-band GX, global maritime",
        "orbit": "GEO",
        "satellites": {
            28628: "INMARSAT 4-F1",
            28899: "INMARSAT 4-F2",
            33278: "INMARSAT 4-F3",
            39476: "INMARSAT 5-F1",
            40384: "INMARSAT 5-F2",
            40882: "INMARSAT 5-F3",
            42698: "INMARSAT 5-F4",
            44801: "INMARSAT GX5",
            50319: "INMARSAT 6-F1",
            55683: "INMARSAT 6-F2",
        },
    },
    "viasat": {
        "desc": "Viasat — Ka-band high-capacity GEO",
        "orbit": "GEO",
        "satellites": {
            37843: "VIASAT-1",
            42740: "VIASAT-2",
            56370: "VIASAT-3 F1",
            66454: "VIASAT-3 F2",
        },
    },
    "thuraya": {
        "desc": "Thuraya — L-band regional (Middle East, Asia, Africa)",
        "orbit": "GEO",
        "satellites": {
            27825: "THURAYA-2",
            32404: "THURAYA-3",
            62483: "THURAYA-4",
        },
    },
    "ses": {
        "desc": "SES — Ku/Ka GEO, capacity for KVH/Marlink/Speedcast/Castor",
        "orbit": "GEO",
        "satellites": {
            34941: "SES-7",
            36516: "SES-1",
            37748: "SES-3",
            37809: "SES-2",
            38087: "SES-4",
            38652: "SES-5",
            39172: "SES-6",
            39460: "SES-8",
            41380: "SES-9",
            42432: "SES-10",
            42967: "SES-11",
            43175: "SES-14",
            43488: "SES-12",
            42709: "SES-15",
            49332: "SES-17",
            55970: "SES-18",
            52933: "SES-22",
        },
    },
    "o3b": {
        "desc": "SES O3b mPOWER — MEO Ka-band, low-latency maritime",
        "orbit": "MEO",
        "celestrak_group": {"NAME": "O3B"},
    },
    "intelsat": {
        "desc": "Intelsat — Ku/C GEO, backbone for Marlink/Speedcast/Anuvu/Globecomm",
        "orbit": "GEO",
        "celestrak_group": {"NAME": "INTELSAT"},
    },
    "eutelsat": {
        "desc": "Eutelsat — Ku GEO, European + African maritime",
        "orbit": "GEO",
        "celestrak_group": {"NAME": "EUTELSAT"},
    },
    "iridium": {
        "desc": "Iridium NEXT — L-band LEO, global including polar",
        "orbit": "LEO",
        "celestrak_group": {"GROUP": "iridium-NEXT"},
    },
    # --- Regional satellite operators ---
    "yahsat": {
        "desc": "Yahsat (Al Yah) — Ka/Ku GEO, Middle East/Africa (IEC Telecom uses)",
        "orbit": "GEO",
        "satellites": {
            37393: "YAHSAT 1A",
            38245: "YAHSAT 1B",
        },
    },
    "jcsat": {
        "desc": "JCSAT (SKY Perfect JSAT) — Ku/Ka GEO, Asia-Pacific maritime",
        "orbit": "GEO",
        "celestrak_group": {"NAME": "JCSAT"},
    },
    "asiasat": {
        "desc": "AsiaSat — Ku/C GEO, Asia-Pacific (Speedcast/Singtel uses)",
        "orbit": "GEO",
        "celestrak_group": {"NAME": "ASIASAT"},
    },
    "measat": {
        "desc": "MEASAT — Ku/C GEO, SE Asia + Indian Ocean maritime",
        "orbit": "GEO",
        "celestrak_group": {"NAME": "MEASAT"},
    },
    "hylas": {
        "desc": "Avanti (HYLAS) — Ka GEO, European + African maritime",
        "orbit": "GEO",
        "satellites": {
            37237: "HYLAS 1",
            38741: "HYLAS 2",
            43272: "HYLAS 4",
        },
    },
}

# ---------------------------------------------------------------------------
# Service provider -> satellite operator mapping
#
# Service providers lease capacity; they don't own satellites.
# This maps which operators' satellites a provider's traffic may traverse.
# ---------------------------------------------------------------------------

SERVICE_PROVIDERS = {
    "kvh":        {"operators": ["ses", "intelsat", "eutelsat"], "desc": "KVH mini-VSAT Broadband"},
    "marlink":    {"operators": ["intelsat", "ses", "eutelsat"], "desc": "Marlink managed VSAT"},
    "speedcast":  {"operators": ["intelsat", "ses", "asiasat"], "desc": "Speedcast maritime + energy"},
    "mtnsat":     {"operators": ["intelsat", "ses"], "desc": "MTNSAT/Anuvu — cruise + commercial"},
    "anuvu":      {"operators": ["intelsat", "ses"], "desc": "Anuvu (ex-MTN) — cruise + commercial"},
    "nsslglobal": {"operators": ["intelsat", "ses", "eutelsat"], "desc": "NSSLGlobal — German maritime"},
    "singtel":    {"operators": ["ses", "intelsat", "asiasat", "measat"], "desc": "Singtel Satellite — APAC"},
    "navarino":   {"operators": ["ses", "intelsat", "eutelsat"], "desc": "Navarino Telecom — Greek maritime"},
    "castor":     {"operators": ["ses", "eutelsat"], "desc": "Castor Marine — Netherlands teleport"},
    "globecomm":  {"operators": ["intelsat", "ses"], "desc": "Globecomm — EU + US maritime"},
    "iec_telecom": {"operators": ["yahsat", "thuraya", "ses"], "desc": "IEC Telecom — Middle East maritime"},
}

# Cache TTL per orbit type
CACHE_TTL_ORBIT = {
    "GEO": 7 * 86400,
    "MEO": 3 * 86400,
    "LEO": 1 * 86400,
}

# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def datetime_to_jd(dt):
    """Convert a UTC datetime to Julian date as (jd, fr) pair for sgp4."""
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond / 1e6)


def gmst_rad(dt):
    """Greenwich Mean Sidereal Time in radians (IAU 1982)."""
    jd, fr = datetime_to_jd(dt)
    jd_full = jd + fr
    T = (jd_full - 2_451_545.0) / 36_525.0
    gmst_sec = (67310.54841
                + (876_600 * 3600 + 8_640_184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    return (gmst_sec % 86400) / 86400 * 2 * pi


def teme_to_ecef(x, y, z, dt):
    """Rotate TEME position to ECEF using GMST."""
    g = gmst_rad(dt)
    cg, sg = cos(g), sin(g)
    return (x * cg + y * sg,
            -x * sg + y * cg,
            z)


def ecef_to_geodetic(x, y, z):
    """ECEF (km) to geodetic (deg, deg, km) via Bowring's method."""
    lon = atan2(y, x)
    p = sqrt(x ** 2 + y ** 2)
    theta = atan2(z * WGS84_A, p * WGS84_B)
    lat = atan2(
        z + (WGS84_E2 / (1 - WGS84_E2)) * WGS84_B * sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * cos(theta) ** 3
    )
    sin_lat = sin(lat)
    N = WGS84_A / sqrt(1 - WGS84_E2 * sin_lat ** 2)
    alt = p / cos(lat) - N if abs(cos(lat)) > 1e-10 else abs(z) - WGS84_B
    return np.degrees(lat), np.degrees(lon), alt


def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """Geodetic (deg, deg, km) to ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt_km) * cos_lat * np.cos(lon)
    y = (N + alt_km) * cos_lat * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_km) * sin_lat
    return np.array([x, y, z])


def compute_slant_range(obs_lat, obs_lon, obs_alt, sat_lat, sat_lon, sat_alt):
    """
    Slant range, elevation, azimuth, and propagation delay
    from an observer to a satellite. Lat/lon in degrees, alt in km.
    """
    obs_ecef = geodetic_to_ecef(obs_lat, obs_lon, obs_alt)
    sat_ecef = geodetic_to_ecef(sat_lat, sat_lon, sat_alt)
    dx = sat_ecef - obs_ecef

    lat_r = np.radians(obs_lat)
    lon_r = np.radians(obs_lon)
    sl, cl = np.sin(lat_r), np.cos(lat_r)
    sn, cn = np.sin(lon_r), np.cos(lon_r)

    e = -sn * dx[0] + cn * dx[1]
    n = -sl * cn * dx[0] - sl * sn * dx[1] + cl * dx[2]
    u = cl * cn * dx[0] + cl * sn * dx[1] + sl * dx[2]

    range_km = float(np.sqrt(e ** 2 + n ** 2 + u ** 2))
    elev = float(np.degrees(np.arctan2(u, np.sqrt(e ** 2 + n ** 2))))
    az = float(np.degrees(np.arctan2(e, n))) % 360

    return {
        "slant_range_km": range_km,
        "elevation_deg": elev,
        "azimuth_deg": az,
        "one_way_delay_ms": range_km / C_KM_S * 1000,
        "round_trip_delay_ms": 2 * range_km / C_KM_S * 1000,
    }


def classify_orbit(mean_motion):
    """Classify orbit from mean motion (rev/day)."""
    if mean_motion < 1.5:
        return "GEO"
    elif mean_motion < 6.0:
        return "MEO"
    else:
        return "LEO"


# ---------------------------------------------------------------------------
# TLE Cache
# ---------------------------------------------------------------------------

class TLECache:
    """Fetch and cache satellite OMM data from Celestrak."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / "sat_tracker_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}

    def _save_index(self):
        self.index_file.write_text(json.dumps(self.index, indent=2))

    def _is_fresh(self, key, orbit_type="GEO"):
        if key not in self.index:
            return False
        fetched = datetime.fromisoformat(self.index[key]["fetched_at"])
        ttl = CACHE_TTL_ORBIT.get(orbit_type, 86400)
        return (datetime.now(timezone.utc) - fetched).total_seconds() < ttl

    def fetch_group(self, key, params, orbit_type="GEO", force=False):
        """Fetch OMM JSON for a Celestrak query. Returns list of dicts."""
        cache_file = self.cache_dir / f"{key}.json"

        if not force and self._is_fresh(key, orbit_type) and cache_file.exists():
            return json.loads(cache_file.read_text())

        query = {**params, "FORMAT": "json"}
        try:
            r = requests.get(CELESTRAK_BASE, params=query, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            if cache_file.exists():
                print(f"  Warning: Celestrak fetch failed ({exc}), using stale cache")
                return json.loads(cache_file.read_text())
            raise

        cache_file.write_text(json.dumps(data, indent=1))
        self.index[key] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(data),
        }
        self._save_index()
        return data

    def fetch_by_norad_id(self, norad_id):
        """Fetch a single satellite by NORAD catalog number."""
        r = requests.get(CELESTRAK_BASE, params={"CATNR": norad_id, "FORMAT": "json"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None

    def search(self, query):
        """Search Celestrak by name."""
        r = requests.get(CELESTRAK_BASE, params={"NAME": query, "FORMAT": "json"}, timeout=15)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class SatellitePropagator:
    """SGP4 propagation engine for VSAT satellites."""

    def __init__(self, cache_dir=None):
        self.cache = TLECache(cache_dir)
        self._satellites = {}   # norad_id -> (Satrec, omm_dict, provider)

    def load_provider(self, provider_name, force_refresh=False):
        """Load all satellites for a VSAT provider."""
        provider_name = provider_name.lower()
        info = VSAT_FLEET.get(provider_name)
        if not info:
            raise ValueError(f"Unknown provider: {provider_name}. "
                             f"Available: {', '.join(VSAT_FLEET.keys())}")

        orbit_type = info["orbit"]

        # Group-fetched providers (Intelsat, Eutelsat, O3b, Iridium)
        if "celestrak_group" in info:
            data = self.cache.fetch_group(
                provider_name, info["celestrak_group"], orbit_type, force=force_refresh)
            count = 0
            for entry in data:
                sat = Satrec()
                try:
                    omm.initialize(sat, entry)
                    self._satellites[sat.satnum] = (sat, entry, provider_name)
                    count += 1
                except Exception:
                    pass
            return count

        # Individual-satellite providers (Inmarsat, Viasat, Thuraya, SES)
        count = 0
        for norad_id, name in info["satellites"].items():
            if norad_id in self._satellites:
                count += 1
                continue
            try:
                entry = self.cache.fetch_by_norad_id(norad_id)
                if entry:
                    sat = Satrec()
                    omm.initialize(sat, entry)
                    self._satellites[sat.satnum] = (sat, entry, provider_name)
                    count += 1
            except Exception:
                pass
        return count

    def load_fleet(self, force_refresh=False):
        """Load all VSAT provider satellites."""
        total = 0
        for name in VSAT_FLEET:
            n = self.load_provider(name, force_refresh)
            total += n
        return total

    def _ensure_loaded(self, norad_id):
        """If satellite not loaded, fetch it individually."""
        if norad_id not in self._satellites:
            entry = self.cache.fetch_by_norad_id(norad_id)
            if entry is None:
                raise ValueError(f"Satellite NORAD {norad_id} not found")
            sat = Satrec()
            omm.initialize(sat, entry)
            self._satellites[sat.satnum] = (sat, entry, "unknown")

    def propagate(self, norad_id, dt=None):
        """Propagate a satellite to a given UTC datetime."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        self._ensure_loaded(norad_id)
        sat, entry, provider = self._satellites[norad_id]

        jd_val, fr_val = datetime_to_jd(dt)
        error, r_teme, v_teme = sat.sgp4(jd_val, fr_val)

        if error != 0:
            return {"norad_id": norad_id, "name": entry.get("OBJECT_NAME", ""),
                    "error": error, "error_msg": f"SGP4 error code {error}"}

        x, y, z = teme_to_ecef(r_teme[0], r_teme[1], r_teme[2], dt)
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        mm = float(entry.get("MEAN_MOTION", 0))

        return {
            "norad_id": norad_id,
            "name": entry.get("OBJECT_NAME", ""),
            "provider": provider,
            "epoch": entry.get("EPOCH", ""),
            "time": dt.isoformat(),
            "position_teme_km": list(r_teme),
            "velocity_teme_km_s": list(v_teme),
            "lat_deg": round(lat, 4),
            "lon_deg": round(lon, 4),
            "alt_km": round(alt, 2),
            "orbit_type": classify_orbit(mm),
            "mean_motion": mm,
            "error": 0,
        }

    def propagate_provider(self, provider_name, dt=None):
        """Propagate all satellites for a provider."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        self.load_provider(provider_name)
        pname = provider_name.lower()
        results = []
        for nid, (sat, entry, prov) in self._satellites.items():
            if prov == pname:
                result = self.propagate(nid, dt)
                if result.get("error", 0) == 0:
                    results.append(result)
        results.sort(key=lambda r: r["lon_deg"])
        return results

    def propagate_fleet(self, dt=None):
        """Propagate all VSAT satellites."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        self.load_fleet()
        results = []
        for nid in self._satellites:
            result = self.propagate(nid, dt)
            if result.get("error", 0) == 0:
                results.append(result)
        results.sort(key=lambda r: (r["provider"], r["lon_deg"]))
        return results

    def propagate_time_series(self, norad_id, start, end, step_seconds=60, observer=None):
        """Propagate a satellite over a time window."""
        self._ensure_loaded(norad_id)
        results = []
        dt = start
        while dt <= end:
            pos = self.propagate(norad_id, dt)
            if pos.get("error", 0) == 0 and observer:
                rng = compute_slant_range(
                    observer[0], observer[1], observer[2],
                    pos["lat_deg"], pos["lon_deg"], pos["alt_km"]
                )
                pos.update(rng)
            results.append(pos)
            dt += timedelta(seconds=step_seconds)
        return results

    def find_satellite(self, query):
        """Search loaded satellites by name substring or NORAD ID."""
        query_upper = query.upper()
        matches = []
        try:
            nid = int(query)
            if nid in self._satellites:
                _, entry, prov = self._satellites[nid]
                matches.append({**entry, "_provider": prov})
                return matches
        except ValueError:
            pass

        for nid, (sat, entry, prov) in self._satellites.items():
            if query_upper in entry.get("OBJECT_NAME", "").upper():
                matches.append({**entry, "_provider": prov})
        return matches


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_time(s):
    if s is None:
        return datetime.now(timezone.utc)
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_observer(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) == 2:
        parts.append(0.0)
    return parts[0], parts[1], parts[2]


def resolve_satellite(prop, args):
    """Resolve --norad or --name to a NORAD ID."""
    if hasattr(args, "norad") and args.norad:
        return args.norad
    if hasattr(args, "name") and args.name:
        matches = prop.find_satellite(args.name)
        if not matches:
            matches = prop.cache.search(args.name)
        if not matches:
            print(f"No satellite found matching '{args.name}'")
            sys.exit(1)
        if len(matches) == 1:
            return int(matches[0]["NORAD_CAT_ID"])
        print(f"Multiple matches for '{args.name}':")
        for m in matches[:20]:
            print(f"  NORAD {m['NORAD_CAT_ID']:>7}  {m['OBJECT_NAME']}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more")
        print("Use --norad to select one.")
        sys.exit(1)
    print("Specify --norad or --name")
    sys.exit(1)


def fmt_lat(deg):
    return f"{abs(deg):.4f} {'N' if deg >= 0 else 'S'}"


def fmt_lon(deg):
    return f"{abs(deg):.4f} {'E' if deg >= 0 else 'W'}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fleet(args):
    prop = SatellitePropagator()
    dt = parse_time(args.time)
    print(f"\nLoading VSAT fleet...")
    prop.load_fleet()
    print(f"Propagating to {dt.isoformat()}...\n")

    current_provider = None
    results = prop.propagate_fleet(dt)

    for r in results:
        if r["provider"] != current_provider:
            current_provider = r["provider"]
            info = VSAT_FLEET.get(current_provider, {})
            print(f"\n--- {current_provider.upper()} ({info.get('orbit', '?')}) "
                  f"— {info.get('desc', '')} ---")
            print(f"  {'Name':<30} {'NORAD':>7}  {'Lat':>10}  {'Lon':>10}  {'Alt (km)':>10}")
            print("  " + "-" * 75)
        print(f"  {r['name']:<30} {r['norad_id']:>7}  {fmt_lat(r['lat_deg']):>10}  "
              f"{fmt_lon(r['lon_deg']):>10}  {r['alt_km']:>10,.1f}")

    print(f"\nTotal: {len(results)} VSAT satellites across {len(VSAT_FLEET)} providers")


def cmd_provider(args):
    prop = SatellitePropagator()
    pname = args.provider.lower()
    dt = parse_time(args.time)

    # Check if it's a service provider (maps to satellite operators)
    if pname in SERVICE_PROVIDERS:
        sp = SERVICE_PROVIDERS[pname]
        print(f"\n{pname.upper()} — {sp['desc']}")
        print(f"Service provider — leases capacity from: {', '.join(o.upper() for o in sp['operators'])}")
        print(f"Propagating to {dt.isoformat()}...\n")

        for op_name in sp["operators"]:
            info = VSAT_FLEET[op_name]
            results = prop.propagate_provider(op_name, dt)
            print(f"\n  --- {op_name.upper()} ({info['orbit']}) ---")
            print(f"  {'Name':<30} {'NORAD':>7}  {'Lat':>10}  {'Lon':>10}  {'Alt (km)':>10}")
            print("  " + "-" * 75)
            for r in results:
                print(f"  {r['name']:<30} {r['norad_id']:>7}  {fmt_lat(r['lat_deg']):>10}  "
                      f"{fmt_lon(r['lon_deg']):>10}  {r['alt_km']:>10,.1f}")
        return

    # Direct satellite operator
    if pname not in VSAT_FLEET:
        print(f"Unknown provider: {pname}")
        print(f"Satellite operators: {', '.join(VSAT_FLEET.keys())}")
        print(f"Service providers:   {', '.join(SERVICE_PROVIDERS.keys())}")
        sys.exit(1)

    info = VSAT_FLEET[pname]
    print(f"\n{pname.upper()} — {info['desc']}")
    print(f"Propagating to {dt.isoformat()}...\n")

    results = prop.propagate_provider(pname, dt)

    print(f"{'Name':<30} {'NORAD':>7}  {'Lat':>10}  {'Lon':>10}  {'Alt (km)':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['norad_id']:>7}  {fmt_lat(r['lat_deg']):>10}  "
              f"{fmt_lon(r['lon_deg']):>10}  {r['alt_km']:>10,.1f}")
    print(f"\n{len(results)} satellites")


def cmd_position(args):
    prop = SatellitePropagator()
    # Pre-load fleet so find_satellite has data to search
    prop.load_fleet()
    nid = resolve_satellite(prop, args)
    dt = parse_time(args.time)
    result = prop.propagate(nid, dt)
    if result.get("error", 0) != 0:
        print(f"Propagation error: {result.get('error_msg', result['error'])}")
        sys.exit(1)

    print(f"\n{result['name']} (NORAD {result['norad_id']})")
    print(f"Provider:  {result['provider']}")
    print(f"Time:      {result['time']}")
    print(f"TLE Epoch: {result['epoch']}")
    print(f"")
    print(f"  Latitude:  {fmt_lat(result['lat_deg'])}")
    print(f"  Longitude: {fmt_lon(result['lon_deg'])}")
    print(f"  Altitude:  {result['alt_km']:,.1f} km")
    print(f"  Orbit:     {result['orbit_type']}")


def cmd_range(args):
    prop = SatellitePropagator()
    prop.load_fleet()
    nid = resolve_satellite(prop, args)
    dt = parse_time(args.time)
    obs = parse_observer(args.observer)

    result = prop.propagate(nid, dt)
    if result.get("error", 0) != 0:
        print(f"Propagation error: {result.get('error_msg', result['error'])}")
        sys.exit(1)

    rng = compute_slant_range(obs[0], obs[1], obs[2],
                              result["lat_deg"], result["lon_deg"], result["alt_km"])

    print(f"\n{result['name']} (NORAD {result['norad_id']})")
    print(f"Time: {dt.isoformat()}")
    print(f"")
    print(f"  Satellite:   {fmt_lat(result['lat_deg'])}, {fmt_lon(result['lon_deg'])}, "
          f"{result['alt_km']:,.1f} km")
    print(f"  Observer:    {fmt_lat(obs[0])}, {fmt_lon(obs[1])}, {obs[2]:.1f} km")
    print(f"")
    print(f"  Slant range: {rng['slant_range_km']:,.1f} km")
    print(f"  Elevation:   {rng['elevation_deg']:.2f} deg")
    print(f"  Azimuth:     {rng['azimuth_deg']:.2f} deg")
    print(f"  One-way:     {rng['one_way_delay_ms']:.3f} ms")
    print(f"  Round-trip:  {rng['round_trip_delay_ms']:.3f} ms")


def cmd_track(args):
    prop = SatellitePropagator()
    prop.load_fleet()
    nid = resolve_satellite(prop, args)
    start = parse_time(args.start)
    end = parse_time(args.end)
    obs = parse_observer(args.observer) if args.observer else None

    results = prop.propagate_time_series(nid, start, end, args.step, observer=obs)
    name = results[0]["name"] if results else "?"

    if obs:
        print(f"\n{'Time':<26} {'Lat':>10} {'Lon':>10} {'Alt km':>10} "
              f"{'Range km':>10} {'Elev':>7} {'RTT ms':>9}")
        print("-" * 95)
        for r in results:
            if r.get("error", 0) != 0:
                continue
            t = r["time"][:25]
            print(f"{t:<26} {r['lat_deg']:>10.4f} {r['lon_deg']:>10.4f} "
                  f"{r['alt_km']:>10.1f} {r.get('slant_range_km', 0):>10.1f} "
                  f"{r.get('elevation_deg', 0):>7.2f} {r.get('round_trip_delay_ms', 0):>9.3f}")
    else:
        print(f"\n{'Time':<26} {'Lat':>10} {'Lon':>10} {'Alt km':>10}")
        print("-" * 62)
        for r in results:
            if r.get("error", 0) != 0:
                continue
            t = r["time"][:25]
            print(f"{t:<26} {r['lat_deg']:>10.4f} {r['lon_deg']:>10.4f} {r['alt_km']:>10.1f}")

    print(f"\n{name} (NORAD {nid}) — {len(results)} points, {args.step}s step")


def cmd_search(args):
    print(f"Searching Celestrak for '{args.query}'...")
    cache = TLECache()
    matches = cache.search(args.query)
    if not matches:
        print("No results.")
        return
    print(f"\n{'Name':<35} {'NORAD':>7}  {'Orbit':>5}  {'Epoch':<22}")
    print("-" * 75)
    for m in matches[:50]:
        otype = classify_orbit(float(m.get("MEAN_MOTION", 0)))
        print(f"{m['OBJECT_NAME']:<35} {m['NORAD_CAT_ID']:>7}  {otype:>5}  {m.get('EPOCH', ''):<22}")
    if len(matches) > 50:
        print(f"... and {len(matches) - 50} more")
    print(f"\nTotal: {len(matches)} results")


def cmd_refresh(args):
    prop = SatellitePropagator()
    print("Refreshing all VSAT fleet TLEs...")
    for pname in VSAT_FLEET:
        info = VSAT_FLEET[pname]
        print(f"  {pname}...", end=" ", flush=True)
        n = prop.load_provider(pname, force_refresh=True)
        print(f"{n} satellites")
    print("Done.")


def cmd_status(args):
    cache = TLECache()
    print("\nVSAT Fleet — Satellite Operators:")
    print(f"  {'Operator':<15} {'Orbit':>5} {'Sats':>5}  Description")
    print("  " + "-" * 70)
    for pname, info in VSAT_FLEET.items():
        n = info.get("satellites", {})
        count = len(n) if n else "grp"
        cached = cache.index.get(pname, {})
        if cached:
            count = cached.get("count", count)
        print(f"  {pname:<15} {info['orbit']:>5} {str(count):>5}  {info['desc']}")

    print(f"\nService Providers (lease capacity):")
    print(f"  {'Provider':<15} {'Uses':>40}  Description")
    print("  " + "-" * 75)
    for pname, sp in SERVICE_PROVIDERS.items():
        ops = ", ".join(o.upper() for o in sp["operators"])
        print(f"  {pname:<15} {ops:>40}  {sp['desc']}")

    print(f"\n  Cache: {cache.cache_dir}")
    if cache.index:
        print(f"  {'Key':<15} {'Fetched':>22}  {'Count':>5}")
        print("  " + "-" * 45)
        for key, val in sorted(cache.index.items()):
            fetched = val.get("fetched_at", "?")[:19]
            print(f"  {key:<15} {fetched:>22}  {val.get('count', '?'):>5}")
    else:
        print("  No cached data. Run: python sat_tracker.py refresh")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def cmd_test(args):
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    print("\n=== VSAT Satellite Tracker Self-Tests ===\n")

    # 1: Julian date
    print("[1] Julian date conversion")
    jd_val, fr_val = datetime_to_jd(datetime(2000, 1, 1, 12, 0, 0))
    check("J2000 epoch", abs((jd_val + fr_val) - 2_451_545.0) < 0.001)

    # 2: GMST
    print("[2] GMST range")
    g = gmst_rad(datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc))
    check("GMST in [0, 2pi]", 0 <= g <= 2 * pi)

    # 3: Coordinate round-trip
    print("[3] Coordinate round-trip")
    for lat, lon, alt in [(0, 0, 0), (45, -120, 100), (-33.8, 151.2, 0.05), (89.9, 0, 0)]:
        ecef = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(ecef[0], ecef[1], ecef[2])
        check(f"  ({lat}, {lon}, {alt})",
              abs(lat - lat2) < 0.01 and abs(lon - lon2) < 0.01 and abs(alt - alt2) < 0.1,
              f"got ({lat2:.4f}, {lon2:.4f}, {alt2:.3f})")

    # 4: Subsatellite slant range
    print("[4] Subsatellite slant range")
    rng = compute_slant_range(0, 0, 0, 0, 0, 35786)
    check("Range ~ 35786 km", abs(rng["slant_range_km"] - 35786) < 10)
    check("Elevation ~ 90 deg", abs(rng["elevation_deg"] - 90) < 1)
    check("One-way ~ 119.4 ms", abs(rng["one_way_delay_ms"] - 119.4) < 1)

    # 5: Inmarsat 4-F2 — GEO Atlantic, should be ~55W
    print("[5] Inmarsat 4-F2 (NORAD 28899)")
    try:
        prop = SatellitePropagator()
        result = prop.propagate(28899, datetime.now(timezone.utc))
        check("No error", result.get("error", -1) == 0)
        check("Name correct", "INMARSAT 4-F2" in result.get("name", ""))
        check("GEO altitude", abs(result["alt_km"] - 35786) < 300,
              f"got {result['alt_km']:.1f}")
        check("Near equator", abs(result["lat_deg"]) < 6,
              f"got {result['lat_deg']:.2f}")
        check("Orbit = GEO", result["orbit_type"] == "GEO")
    except Exception as exc:
        print(f"  SKIP  ({exc})")

    # 6: Thuraya-3 — GEO ~98.5E
    print("[6] Thuraya-3 (NORAD 32404)")
    try:
        result = prop.propagate(32404, datetime.now(timezone.utc))
        check("No error", result.get("error", -1) == 0)
        check("GEO altitude", abs(result["alt_km"] - 35786) < 300)
        check("Orbit = GEO", result["orbit_type"] == "GEO")
    except Exception as exc:
        print(f"  SKIP  ({exc})")

    # 7: O3b — MEO ~8062 km
    print("[7] O3b MEO")
    try:
        matches = prop.cache.search("O3B FM2")
        if matches:
            nid = int(matches[0]["NORAD_CAT_ID"])
            result = prop.propagate(nid, datetime.now(timezone.utc))
            check("No error", result.get("error", -1) == 0)
            check("MEO altitude 7000-9000 km", 7000 < result["alt_km"] < 9000,
                  f"got {result['alt_km']:.1f}")
            check("Orbit = MEO", result["orbit_type"] == "MEO")
        else:
            print("  SKIP  (O3B FM2 not found)")
    except Exception as exc:
        print(f"  SKIP  ({exc})")

    # 8: Iridium NEXT — LEO ~780 km
    print("[8] Iridium NEXT")
    try:
        matches = prop.cache.search("IRIDIUM 106")
        if matches:
            nid = int(matches[0]["NORAD_CAT_ID"])
            result = prop.propagate(nid, datetime.now(timezone.utc))
            check("No error", result.get("error", -1) == 0)
            check("LEO altitude 700-900 km", 700 < result["alt_km"] < 900,
                  f"got {result['alt_km']:.1f}")
            check("Orbit = LEO", result["orbit_type"] == "LEO")
        else:
            print("  SKIP  (IRIDIUM 106 not found)")
    except Exception as exc:
        print(f"  SKIP  ({exc})")

    # 9: Slant range — use satellite's actual subsatellite point as observer ref
    print("[9] Slant range validation")
    try:
        result = prop.propagate(28899, datetime.now(timezone.utc))
        if result.get("error", 0) == 0:
            # Observer 30 deg away in longitude — should have clear visibility
            obs_lon = result["lon_deg"] - 30
            rng = compute_slant_range(20, obs_lon, 0,
                                      result["lat_deg"], result["lon_deg"], result["alt_km"])
            check("Range 35000-41000 km", 35000 < rng["slant_range_km"] < 41000,
                  f"got {rng['slant_range_km']:.1f}")
            check("Elevation > 0", rng["elevation_deg"] > 0,
                  f"got {rng['elevation_deg']:.2f}")
            check("RTT 235-280 ms", 235 < rng["round_trip_delay_ms"] < 280,
                  f"got {rng['round_trip_delay_ms']:.3f}")
    except Exception as exc:
        print(f"  SKIP  ({exc})")

    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VSAT Satellite Position Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s fleet                                    # all VSAT sats, current positions
  %(prog)s provider inmarsat                        # one provider's fleet
  %(prog)s position --name "INMARSAT 4-F2"          # single satellite
  %(prog)s position --norad 28899 --time 2026-02-20T14:30:00Z
  %(prog)s range --norad 28899 --observer 25,-80,0  # slant range + delay
  %(prog)s track --norad 28899 --start 2026-02-22T00:00Z --end 2026-02-22T06:00Z
  %(prog)s search "INTELSAT"
  %(prog)s refresh
  %(prog)s status
  %(prog)s test
"""
    )
    sub = parser.add_subparsers(dest="command")

    # fleet
    f = sub.add_parser("fleet", help="Show all VSAT satellite positions")
    f.add_argument("--time", help="UTC time (ISO 8601), default=now")

    # provider
    p = sub.add_parser("provider", help="Show one provider's satellites")
    p.add_argument("provider", help="Satellite operator or service provider name")
    p.add_argument("--time", help="UTC time, default=now")

    # position
    pos = sub.add_parser("position", help="Single satellite position")
    pos.add_argument("--norad", type=int, help="NORAD catalog ID")
    pos.add_argument("--name", help="Satellite name")
    pos.add_argument("--time", help="UTC time, default=now")

    # range
    rng = sub.add_parser("range", help="Slant range from observer")
    rng.add_argument("--norad", type=int)
    rng.add_argument("--name")
    rng.add_argument("--observer", required=True, help="lat,lon,alt_km")
    rng.add_argument("--time")

    # track
    trk = sub.add_parser("track", help="Track satellite over time window")
    trk.add_argument("--norad", type=int)
    trk.add_argument("--name")
    trk.add_argument("--start", required=True)
    trk.add_argument("--end", required=True)
    trk.add_argument("--step", type=float, default=60, help="Step seconds (default 60)")
    trk.add_argument("--observer", help="lat,lon,alt_km for range calc")

    # search
    s = sub.add_parser("search", help="Search Celestrak by name")
    s.add_argument("query")

    # refresh / status / test
    sub.add_parser("refresh", help="Refresh all VSAT fleet TLEs")
    sub.add_parser("status", help="Show fleet and cache status")
    sub.add_parser("test", help="Run self-test validation")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "fleet": cmd_fleet,
        "provider": cmd_provider,
        "position": cmd_position,
        "range": cmd_range,
        "track": cmd_track,
        "search": cmd_search,
        "refresh": cmd_refresh,
        "status": cmd_status,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
