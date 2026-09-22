# Statistical Analysis — River Morphodynamics (Historical Findings)

This folder contains the statistical and spatial analysis of the Padma River's morphological evolution over the 38-year study period (1988–2025), performed **before model training** to characterise baseline river behaviour.

> **Full details** are in the **pre-defence report**: [`docs/predefence_report.md`](docs/predefence_report.md) — see **Chapter 6** (§6.1 *Spatiotemporal Statistical Analysis of Padma River Dynamics*). Methodology details are in [`docs/methodology.md`](docs/methodology.md).

---

## Notebook

| File | Purpose |
|------|---------|
| `results/01_statistical_analysis.ipynb` | Complete morphodynamic analysis: areal dynamics, erosion/accretion mapping, channel migration, Water Occurrence Frequency (WOF), and Markov transition modelling |

---

## Pipeline Overview

```
preprocessed binary water masks (yearly, quarterly, bi-monthly)
        │
        ▼
   1. Areal Dynamics
      • Annual water area time-series (1988–2025)
      • Mann-Kendall trend test + Theil-Sen slope
      • Seasonal CV analysis
      • Quarterly heat-map
        │
        ▼
   2. Erosion & Accretion
      • Pixel-level land↔water transition counting
      • Annual gross erosion / accretion / net change
      • Decadal phase analysis
        │
        ▼
   3. Channel Migration
      • Centreline extraction + distance tracking
      • Annual lateral migration rates (m/yr)
        │
        ▼
   4. Probabilistic Mapping
      • Water Occurrence Frequency (WOF)
      • Markov transition matrix (land↔water)
      • Erosion / accretion / instability risk maps
```

---

## 1. Areal Dynamics and Seasonal Variability  (§6.1.1)

### Annual water-covered area

Over the 38-year record the Padma River showed strong inter-annual variability:

| Statistic | Value |
|-----------|-------|
| Minimum | **540.4 km²** (1995, dry year) |
| Maximum | **1,229.3 km²** (1999, post-flood) |
| Long-term mean | **664.5 ± 112.8 km²** (1σ) |

The 1999 maximum corresponds to the catastrophic monsoon flood; the 1995 minimum reflects a comparatively dry hydrological year.

### Trend analysis

- **Mann–Kendall test:** no statistically significant monotonic trend (τ = −0.149, p = 0.187).
- **Theil–Sen slope estimator:** −1.41 km² yr⁻¹ (weakly negative).

**Conclusion:** despite strong year-to-year fluctuations, the long-term mean river extent has remained broadly stable — no evidence of sustained expansion or contraction over 1988–2025.

### Seasonal variability

The annual coefficient of variation (CV) of bi-monthly water area quantifies flood-pulse intensity:

| CV range | Interpretation | Example years |
|----------|----------------|---------------|
| Low (8–15 %) | Stable seasonal extent, weak flood pulse | 2018 (8.2 %, most stable) |
| High (40–64 %) | Strong inter-seasonal swings, major flood years | 2007 (64.1 %), 2011, 2020 |
| **Mean (all years)** | **34.3 %** | — |

Study-period mean CV = **34.3 %**. High-CV years are associated with monsoon flood events where bi-monthly water extent exceeded 2,700 km² during peak inundation.

### Visualisations produced

- **Figure 6.2:** Bi-monthly water area profiles for all years (coloured by year) + annual CV bar chart.
- **Figure 6.3:** Quarterly heat-map of water area (years × quarters).

### Quarterly pattern

The quarterly heat-map (Figure 6.3) shows a consistent seasonal structure across all years:
- **Q4 (Oct–Dec):** largest water extent (post-monsoon peak)
- **Q2 (Apr–Jun):** minimum extent (pre-monsoon dry phase)

This reflects the dominant influence of the South Asian monsoon on Padma River hydrology.

## 2. Spatial Change Analysis: Erosion & Accretion  (§6.1.2)

### Pixel-level change detection

For each consecutive year pair, pixels are classified as:
- **Erosion:** land→water (0→1)
- **Accretion:** water→land (1→0)
- **Stable water:** 1→1
- **Stable land:** 0→0

### Key historical events

| Event | Year | Erosion (km²) | Accretion (km²) | Net change (km²) |
|-------|------|---------------|-----------------|------------------|
| Largest erosion | 1998–1999 | **649.1** | — | **+567.7** (expansion) |
| Largest accretion | 1999–2000 | — | **503.5** | **−503.5** (contraction) |

The 1998–1999 event represents the most extreme widening recorded; the immediate 1999–2000 reversal highlights the strong post-flood recovery phase.

### Cumulative totals (1988–2025)

| Metric | Value |
|--------|-------|
| Cumulative gross erosion | **5,170.2 km²** |
| Cumulative gross accretion | **5,323.3 km²** |
| **Net change** | **+153.0 km²** (net accretion, i.e. slight channel narrowing) |

### Decadal phases

| Period | Erosion (km²) | Accretion (km²) | Net change (km²) | Interpretation |
|--------|---------------|-----------------|------------------|----------------|
| 1988–1998 | 367.6 | 434.2 | **−66.7** | Near-balanced, moderate lateral migration |
| 1998–2008 | 382.3 | 355.9 | **+26.3** | Slight net erosion, driven by 1998–1999 flood |
| 2008–2018 | 262.8 | 375.8 | **−113.0** | Strongest net accretion, significant channel contraction |
| 2018–2025 | 264.6 | 264.3 | **+0.3** | Near-equilibrium, comparative stability |
| **Total** | **5,170.2** | **5,323.3** | **−153.0** | Slightly narrowing trend overall |

> Note on sign convention: **positive net change = net erosion (expansion)**, **negative = net accretion (contraction)**.

### Visualisations produced

- **Figure 6.4:** Annual erosion/accretion bar chart + decadal net change timeline.

Study-period mean CV = **34.3 %**. High-CV years are associated with monsoon flood events where bi-monthly water extent exceeded 2,700 km² during peak inundation.

### Quarterly pattern

The quarterly heat-map shows a consistent seasonal structure across all years:
- **Q4 (Oct–Dec):** largest water extent (post-monsoon peak)
- **Q2 (Apr–Jun):** minimum extent (pre-monsoon dry phase)

This reflects the dominant influence of the South Asian monsoon on Padma River hydrology.

### Visualisations produced

---

## 3. Morphological Metrics & Channel Migration  (§6.1.3)

### Centreline migration

The river centreline is extracted from each yearly binary water mask. Lateral migration rate is computed as the distance between successive year centreline positions.

| Statistic | Value |
|-----------|-------|
| Mean annual migration | **255.4 m yr⁻¹** |
| Median annual migration | **216.3 m yr⁻¹** (lower than mean → skewed by extremes) |
| Minimum | **134.2 m yr⁻¹** (2001–2002, 2022–2023) |
| Maximum | **1,485.5 m yr⁻¹** (1998–1999, catastrophic flood) |
| Cumulative displacement | **> 9 km** over 38 years |

The 1990s recorded the highest average migration rates — the period of maximum geomorphic instability. Low rates in the early 2000s and recent years indicate phases of temporary channel stabilisation.

### Visualisations produced

- **Figure 6.5:** Annual lateral migration rate of the centreline (bar chart, 1988–2025).

- **Figure 6.1:** Long-term annual water area time-series with Mann-Kendall / Theil-Sen annotations.
- **Figure 6.2:** Bi-monthly water area profiles for all years (coloured by year) + annual CV bar chart.
- **Figure 6.3:** Quarterly heat-map of water area (years × quarters).

---

## 4. Probabilistic Mapping: WOF & Markov  (§6.1.4 / Risk Zone Mapping)

### Water Occurrence Frequency (WOF)

For each pixel, WOF = fraction of all time steps where the pixel is classified as water (value = 1). WOF is a value in [0, 1]:
- **WOF ≈ 1:** permanently inundated (inner channel)
- **WOF ≈ 0:** permanently land (floodplain away from channel)
- **Intermediate WOF:** seasonally/intermittently flooded (channel margins, chars)

### Markov transition matrix

A 2×2 Markov transition matrix is computed from consecutive yearly masks:

| From \ To | Land (0) | Water (1) |
|-----------|----------|-----------|
| **Land (0)** | P(0→0) | P(0→1) = erosion probability |
| **Water (1)** | P(1→0) = accretion probability | P(1→1) |

This quantifies the likelihood of each pixel transitioning between land and water states, providing a probabilistic basis for risk assessment.

### Erosion / Accretion / Instability risk maps

Three raster layers are produced:

1. **Erosion frequency** — fraction of transitions where land→water occurred (red channel). High values = pixels frequently lost to the river.
2. **Accretion frequency** — fraction of transitions where water→land occurred (blue channel). High values = pixels frequently deposited.
3. **Instability index** — `I = 1 − |2·F_w − 1|` where F_w is the water fraction over time. Values near 1 indicate high oscillation between land and water (morphologically active zones); values near 0 indicate stable land or stable water.

These three layers form the basis of the **spatially explicit erosion / accretion / instability risk map** used for disaster preparedness (see also `analysis/02_long_term_prediction.ipynb` for the forecast-era risk map).

### Visualisations produced

- WOF map (single band, [0,1])
- Markov transition matrix (2×2 table / heatmap)
- Three-panel risk figure: erosion frequency + accretion frequency + instability index
- Interactive **Folium** risk map (satellite basemap with overlay layers)

---

## 5. Data Sources

The notebook uses all three temporal resolutions:

| Resolution | Period | Count | Role in analysis |
|------------|--------|-------|-----------------|
| Yearly | 1988–2025 | 38 masks | Primary: areal dynamics, erosion/accretion, migration |
| Bi-monthly (BM1–BM6) | 2015–2025 | 156 usable | Seasonal CV, intra-annual variability |
| Quarterly (Q1–Q4) | 2015–2025 | 125 usable | Quarterly heat-map |

**Note:** 1987 is excluded — imagery was confirmed broken/unusable. The study period begins in 1988.

---

## 6. Environment Variables

The notebook reads data paths from environment variables (same convention as the rest of the repository):

| Variable | Default | Description |
|----------|---------|-------------|
| `YEARLY_DIR` | `data/raw/yearly` | Yearly water-mask GeoTIFFs (required) |
| `QUARTERLY_DIR` | `data/raw/quarterly` | Quarterly water-mask GeoTIFFs (optional, for heat-map) |
| `BIMONTHLY_DIR` | `data/raw/bimonthly` | Bi-monthly water-mask GeoTIFFs (optional, for CV analysis) |

Set `YEARLY_DIR` before running the notebook. See [`docs/repository_guide.md`](docs/repository_guide.md) for details.

---

## 7. Key Findings Summary

| Finding | Value |
|---------|-------|
| Long-term mean annual water area | **664.5 ± 112.8 km²** |
| Annual area range | **540.4 – 1,229.3 km²** |
| Trend (Mann-Kendall) | Not significant (τ = −0.149, p = 0.187) |
| Theil-Sen slope | **−1.41 km² yr⁻¹** (weakly negative) |
| Cumulative gross erosion | **5,170.2 km²** |
| Cumulative gross accretion | **5,323.3 km²** |
| Net change (1988–2025) | **+153.0 km²** (slight net narrowing) |
| Strongest accretion decade | **2008–2018** (−113.0 km²) |
| Largest single-year erosion | **1998–1999** (649.1 km²) |
| Mean centreline migration | **255.4 m yr⁻¹** |
| Maximum migration | **1,485.5 m yr⁻¹** (1998–1999) |
| Cumulative centreline displacement | **> 9 km** |
| Mean seasonal CV | **34.3 %** |

---

## 8. Relationship to the Rest of the Pipeline

```
preprocessing/          results/              models/              analysis/
────────────────›  01_statistical_analysis   ───────›  00_*_model_comparison
01_gap_filling_methods     (baseline stats)        (train models)
                                                │
                                                ▼
                                        analysis/02_long_term_prediction
                                        (2026–2040 forecast + risk map)
```

1. Run `preprocessing/01_gap_filling_methods.ipynb` → gap-filled masks.
2. Run `results/01_statistical_analysis.ipynb` → baseline morphodynamic characterisation (this step is **diagnostic**, not required for model training, but informs interpretation of model results).
3. Train models in `models/`.
4. Run `analysis/02_long_term_prediction.ipynb` → forecast + risk map (uses WOF / instability concepts from the statistical analysis).

---

## 9. References

- Pre-defence report: [`docs/predefence_report.md`](docs/predefence_report.md) — **Chapter 6** (§6.1 *Spatiotemporal Statistical Analysis*).
- Methodology: [`docs/methodology.md`](docs/methodology.md) — §6 *Statistical Metrics*, §7 *Experimental Design*.
- Repository guide: [`docs/repository_guide.md`](docs/repository_guide.md).
- Root README: [`../README.md`](../README.md).
- Visualisation helpers: [`utils/visualization_utils.py`](../utils/visualization_utils.py) — `compute_change_frequencies()`, `plot_risk_maps()`, `create_folium_risk_map()`.