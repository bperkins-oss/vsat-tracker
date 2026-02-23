# VSAT Satellite Tracker

Interactive web map showing real-time positions for 370+ maritime VSAT satellites across 13 operators and 11 service providers.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/bperkins-oss/vsat-tracker)

![Dark themed satellite map with Leaflet](https://img.shields.io/badge/map-Leaflet%20%2B%20CartoDB%20Dark-0d1117?style=flat-square)

## Features

- **Interactive map** — Leaflet + CartoDB dark tiles, color-coded by orbit (orange=GEO, blue=MEO, green=LEO)
- **370+ satellites** — Inmarsat, Viasat, SES, Intelsat, Eutelsat, O3b, Iridium, Thuraya, and more
- **Filter by provider** — operators (SES, Intelsat, ...) and service providers (KVH, Marlink, Speedcast, ...)
- **Orbit type toggles** — show/hide GEO, MEO, LEO independently
- **Search** — find satellites by name or NORAD ID
- **Slant range calculator** — click any satellite to compute range, elevation, azimuth, and round-trip delay from an observer location
- **Auto-refresh** — positions update every 30 seconds
- **Sortable data table** — click column headers to sort

## Satellite Operators

| Operator | Orbit | Coverage |
|----------|-------|----------|
| Inmarsat | GEO | Global L-band + Ka (GX) |
| Viasat | GEO | Ka high-capacity |
| SES | GEO | Global Ku/Ka |
| Intelsat | GEO | Global Ku/C backbone |
| Eutelsat | GEO | Europe + Africa Ku |
| O3b (SES) | MEO | Low-latency Ka equatorial |
| Iridium NEXT | LEO | Global L-band incl. polar |
| Thuraya | GEO | Middle East, Asia, Africa |
| Yahsat | GEO | Middle East/Africa Ka |
| JCSAT | GEO | Asia-Pacific Ku/Ka |
| AsiaSat | GEO | Asia-Pacific Ku/C |
| MEASAT | GEO | SE Asia + Indian Ocean |
| HYLAS | GEO | Europe + Africa Ka |

## Service Providers

KVH, Marlink, Speedcast, MTNSAT/Anuvu, NSSLGlobal, Singtel, Navarino, Castor, Globecomm, IEC Telecom — each mapped to the satellite operators they lease capacity from.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check + satellite count |
| `GET /api/fleet?time=` | All satellite positions (default: now) |
| `GET /api/providers` | Operator + service provider list |
| `GET /api/provider/{name}` | One operator or service provider's satellites |
| `GET /api/satellite/{norad_id}?time=` | Single satellite position |
| `GET /api/range?norad_id=&lat=&lon=&alt=` | Slant range + RTT from observer |
| `GET /api/search?q=` | Search by name or NORAD ID |

All position endpoints accept an optional `time` parameter (ISO 8601 UTC). Default is current time.

## Run Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000. First load fetches TLEs from Celestrak (~15-30s), then all positions are computed from cached in-memory Satrec objects.

## Deploy

**Render** (free tier): Click the deploy button above, or connect the repo at https://dashboard.render.com.

**Railway:**
```bash
railway init
railway up
```

**Any platform** — just needs Python 3.10+ and the `Procfile`:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Architecture

Single FastAPI app, no database, no auth, no build step.

- `app.py` — FastAPI routes wrapping `SatellitePropagator`
- `sat_tracker.py` — SGP4 propagation engine, TLE cache, coordinate transforms
- `static/index.html` — single-page frontend (Leaflet map, vanilla JS)

On startup, the lifespan handler preloads all VSAT TLEs from Celestrak into memory. After that, every position request is pure math (SGP4 propagation + TEME-to-geodetic conversion) with zero network calls.

## Tech Stack

- **Backend:** FastAPI, sgp4, NumPy
- **Frontend:** Leaflet.js, CartoDB dark tiles, vanilla JS
- **Data:** Celestrak OMM/JSON API
