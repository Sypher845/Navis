# Navis

**Navis — Event-Driven Congestion Decision Platform for Bengaluru Traffic Police**

> How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?

## What It Does

Navis takes **8,173 real traffic incidents** from Bengaluru's ASTraM system and builds a decision intelligence layer that tells traffic managers **exactly what to do** when an event hits — and **why**.

It's not a dashboard that shows charts. It's an **operational advisory system** that outputs actionable decisions with a transparent reasoning trail.

### Core Capabilities

| Capability | How It Works |
|---|---|
| **Impact Forecasting** | Statistical analysis of historical incidents within concentric radius zones around an event location. Severity scores computed from density, cause similarity, time-of-day match, and priority distribution — all derived from data, not assumptions. |
| **ML Decision Engine** | An XGBoost classifier trained on 6,538 incidents (99.9% test accuracy) with SMOTE class balancing. Outputs one of 4 action classes: `monitor`, `deploy_light`, `deploy_heavy`, `full_closure`. Every recommendation comes with a step-by-step reasoning trail powered by SHAP tree explainer weights. |
| **Junction Barricading** | Uses 224 named junctions from the dataset (SilkBoardJunc, HebbalFlyover, MekhriCircle, etc.) to recommend barricade placement at actual control points — not arbitrary map coordinates. Each junction is scored by incident count, high-priority rate, and historical closure rate. |
| **Time-Machine Simulator** | Replays the most chaotic gridlock day from the dataset (December 16, 2023) sequentially via an interactive slider, feeding the historical sequence natively into the XGBoost engine to demonstrate real-time AI reactions to cascading failure. |
| **Data-Driven Diversions** | Built a corridor co-occurrence matrix from 21 corridors across 152 days. Recommends diversion routes that are historically **least co-affected** with the congested corridor, and shows the independence percentage. |
| **Police Station Routing** | Identifies the 4 nearest police stations from 54 in the dataset, with historical case load and response metrics, so dispatchers know exactly who to call. |

### What Makes This Different

1. **No hardcoded heuristics.** Every number you see (closure probability, incident density normalization, dataset day count) is computed from the actual data at runtime. If the dataset changes, the system adapts.

2. **Transparent reasoning.** The ML model isn't a black box. Every recommendation shows the exact decision path via global SHAP explainability weights: "Driven by Priority (31% influence) → Driven by Road Closure (34% influence) → **DEPLOY HEAVY** (99% confidence, 50 historical cases)".

3. **Operationally specific.** Instead of "deploy 10 police" (from where?), we say "Deploy from Cubbon Park PS (0.49km, 212 past cases) and Halasuru Gate PS (1.03km, 297 past cases). Barricade at AnepalyaJunc and Richmond Circle Jn."

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser (Vanilla JS)              │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ data/    │──▶│ engine/  │──▶│  UI Rendering   │ │
│  │ index.js │   │          │   │  (main.js)       │ │
│  │          │   │ forecast │   │                  │ │
│  │ In-memory│   │ decision │   │ Map + Charts +   │ │
│  │ query    │   │ resource │   │ Reasoning Trail  │ │
│  │ engine   │   │ simulator│   │ Time-Machine     │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
└───────────────────────┬─────────────────────────────┘
                        │ import
              ┌─────────┴──────────┐
              │  exports/          │
              │  baseline_features.json
              │  explainability_weights.json
              │  simulation_playback.json
              │  model_logic.js    │ ← compile_model.py
              └────────────────────┘
```

### Data Pipeline

1. **`scripts/preprocess.py`** — Converts raw ASTraM CSV (45 columns, 8,173 rows) into optimized JSON with temporal features (hour, day-of-week), resolution times, and junction/police station metadata.

2. **`scripts/compile_model.py`** — The master Offline Compiler pipeline. It handles Temporal/Spatial feature engineering (Phase 1), SMOTE balancing and XGBoost training (Phase 2), `m2cgen` JS compilation + SHAP calculation (Phase 3), and simulation sequence extraction (Phase 4).

3. **Browser-side inference** — The compiled `model_logic.js` ES Module is imported directly into the browser and called natively as a JS function. Zero dependencies, no server required for inference.

### Engine Modules

| Module | Purpose |
|---|---|
| `forecast.js` | Statistical impact prediction from spatial-temporal incident patterns |
| `decision.js` | XGBoost wrapper with SHAP reasoning trail reconstruction |
| `resource.js` | Manpower, barricade, diversion, equipment, and police station recommendations |
| `simulator.js`| Time-Machine playback engine orchestrator |
| `learning.js` | Post-event analysis with spike detection and effectiveness scoring |

## Tech Stack

- **Frontend:** Vanilla JavaScript (no React/Vue)
- **Bundler:** Vite
- **Map:** Leaflet + CartoDB dark tiles (Mappls-ready — just set API key in `config.js`)
- **Charts:** Chart.js
- **ML:** XGBoost + SMOTE (Python training pipeline), compiled to JS via `m2cgen`
- **Data:** ASTraM Bengaluru incident data (Nov 2023 — Apr 2024)

## Setup

```bash
# Install dependencies
npm install

# Preprocess the raw CSV into optimized JSON
python scripts/preprocess.py

# Train the XGBoost Decision Engine (exports pure JS + JSON to exports/ folder)
python scripts/compile_model.py

# Start the dev server
npm run dev
```

Open `http://localhost:5173` and:
1. Click on the map to select an event location
2. Choose event type, hour, day
3. Click **Predict Impact**
4. View the ML decision, reasoning trail, junction barricades, police stations, and diversions

## Model Performance

| Metric | Value |
|---|---|
| Training set | 6,538 incidents (Augmented to 15,124 via SMOTE) |
| Test set | 1,635 incidents |
| Overall accuracy | 99.9% |
| `monitor` recall | 99.8% |
| `deploy_light` recall | 100.0% |
| `deploy_heavy` recall | 100.0% |
| `full_closure` recall | 95.8% |

SMOTE (Synthetic Minority Over-sampling Technique) was applied during the Offline Compilation phase to address severe class imbalances, allowing the model to achieve near-perfect recall for rare but critical events like `deploy_heavy` and `full_closure`.

## Data Source

**ASTraM (Advanced Traffic Management System)** — Bengaluru Traffic Police  
8,173 anonymized incident records across 152 days (Nov 2023 – Apr 2024), covering 21 corridors, 224 junctions, and 54 police stations.
