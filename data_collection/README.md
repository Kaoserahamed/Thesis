# Data Collection — Google Earth Engine Water Mask Export

This folder contains three notebooks that export **binary water masks** for the Padma River, Bangladesh from Google Earth Engine (GEE) as 60 m GeoTIFFs (EPSG:4326).

| Notebook | Resolution | Period | Sensors | Output |
|----------|-----------|--------|---------|--------|
| `01_yearly_data_collection.ipynb` | Yearly composite | 1988–2025 (38 years) | Landsat 5/7/8 (1988–2014) + Sentinel-2 (2015–2025) | `Yearly_WaterMasks_1988_2025/` |
| `02_bimonthly_data_collection.ipynb` | Bi-monthly composite (6 per year: BM1–BM6) | 2015–2025 (66 periods) | Sentinel-2 + Sentinel-1 fallback | `Sentinel_BiMonthly_WaterMasks_2015-2025/` |
| `03_quarterly_data_collection.ipynb` | Quarterly composite (4 per year: Q1–Q4) | 2015–2025 (44 quarters) | Sentinel-2 + Sentinel-1 fallback | `Quarterly_WaterMasks_2015-2025/` |

> **Full technical details** (sensor selection logic, band renaming, cloud masking bits, radiometric scaling, SAR fallback threshold, image availability statistics) are in the **pre-defence report**: [`docs/predefence_report.md`](docs/predefence_report.md) — see §4.2 *Data Acquisition* and §4.3 *Data Preprocessing*.

---

## 1. Google Cloud Console — Account & Project Setup

### 1.1 Create a Google account

If you do not already have one, create a Google account at <https://accounts.google.com>.

### 1.2 Open Google Cloud Console

Go to <https://console.cloud.google.com> and sign in with your Google account.

### 1.3 Create a new project

1. In the top navigation bar, click the project selector dropdown (next to the Google Cloud logo).
2. Click **NEW PROJECT**.
3. **Project name** — enter a descriptive name, e.g. `padma-river-water-masks`.
4. **Location** — choose your organisation (or leave as *No organisation* for a personal project).
5. Click **CREATE**.
6. Wait for the project to be created, then select it from the project dropdown.

### 1.4 Enable the Earth Engine API

1. With your project selected, go to **APIs & Services → Library**.
2. Search for **Earth Engine API**.
3. Click it and press **ENABLE**.

### 1.5 Verify your identity (one-time)

When you first use the Python API, `ee.Authenticate()` will open a link. Follow it to grant your Google account access to Earth Engine.

---

## 2. Install the Required Packages

The notebooks require `earthengine-api` and `geemap`.

```bash
pip install earthengine-api geemap
```

If you are on Google Colab the packages are pre-installed; just run the cells.

---

## 3. Authentication & Project ID

Every notebook in this folder initialises Earth Engine with:

```python
ee.Initialize(project='river-468515')
```

The project ID `river-468515` is the one used for the original thesis runs. **Replace it with your own project ID** if you created a different project in §1.3. The project ID appears in the Google Cloud Console URL:

```
https://console.cloud.google.com/home/dashboard?project=YOUR_PROJECT_ID
```

**Do not commit your personal project ID to version control.** If you prefer, store it in an environment variable and read it at runtime:

```bash
# Windows (Command Prompt)
set GEE_PROJECT=river-468515

# Windows (PowerShell)
$env:GEE_PROJECT="river-468515"

# Linux/macOS
export GEE_PROJECT=river-468515
```

Then in the notebook:

---

## 4. Step-by-Step: Running a Data Collection Notebook

The procedure is identical for all three notebooks. Below we use the yearly notebook as the example; substitute the file name for quarterly or bi-monthly as needed.

### Step 1 — Open the notebook

```bash
jupyter notebook data_collection/01_yearly_data_collection.ipynb
```

Or open it directly in Google Colab.

### Step 2 — Authenticate and initialise (first cell)

The first code cell calls `ee.Authenticate()` (only needed once per machine) and then `ee.Initialize(project='river-468515')`.

- On **Colab**: the authentication flow is handled inline.
- On a **local Jupyter** server: a browser window will open; sign in with the Google account linked to your GEE project and paste the authorisation code back into the cell.

```python
import os
PROJECT_ID = os.environ.get("GEE_PROJECT", "river-468515")
ee.Initialize(project=PROJECT_ID)
```

### Step 3 — Review the configuration cell

Each notebook has a clearly marked **CONFIGURATION** block. Key parameters:

| Parameter | Yearly | Quarterly | Bi-monthly |
|-----------|--------|-----------|------------|
| `START_YEAR` | 1988 | 2015 | 2015 |
| `END_YEAR` | 2025 | 2025 | 2025 |
| `out_dir` | `./Yearly_WaterMasks_1988_2025` | `./Quarterly_WaterMasks_2015-2025` | `./Sentinel_BiMonthly_WaterMasks_2015-2025` |
| `EXPORT_SCALE` | 60 m | 60 m | 60 m |
| `CLOUD_COVER_MAX` | 50 % | 50 % | 50 % |
| `WATER_THRESHOLD` | MNDWI > 0 | MNDWI > 0 | MNDWI > 0 |
| `MAX_S2_IMAGES` | — | 25 | 25 |
| `MAX_S1_IMAGES` | — | 20 | 20 |

**Study area polygon** (Padma River, mid-course section):

```python
study_area = ee.Geometry.Polygon([
    [88.75677098777692, 24.00347653384235],   # NW
    [90.59972753074567, 23.152644569433964],   # SE
    [90.52556981590192, 23.54602121781216],    # NE
    [88.74480244337559, 24.334092333430064],   # SW
    [88.75677098777692, 24.00347653384235],    # close ring
])
```

Modify `out_dir` if you want the GeoTIFFs written somewhere else.

### Step 4 — Run the helper-function cells

These define the cloud-masking functions (`mask_landsat_clouds`, `mask_sentinel2_clouds`), the MNDWI computation, the SAR fallback, the composite builder, and the export/download helper. Run them in order.

**Water detection logic (same across all three notebooks):**

- **Primary:** MNDWI = (Green − SWIR1) / (Green + SWIR1); pixel is water when MNDWI > 0.
  - Landsat 5/7: Green = `SR_B2`, SWIR1 = `SR_B5`
  - Landsat 8: Green = `SR_B3`, SWIR1 = `SR_B6`
  - Sentinel-2: Green = `B3`, SWIR1 = `B11`
- **Fallback (Sentinel-1 SAR):** when no Sentinel-2 scene passes the cloud filter, water = VV backscatter < −16 dB.

**Cloud masking:**

- Landsat: `QA_PIXEL` bit 3 (cloud shadow) + bit 4 (cloud).
- Sentinel-2: `QA60` bit 10 (opaque cloud) + bit 11 (cirrus).

### Step 5 — Run the export loop

The main loop iterates over every year (yearly) or every quarter/bi-monthly period (quarterly/bi-monthly). For each period it:

1. Filters the image collection by date, bounds, and cloud cover.
2. Sorts by cloud fraction (ascending) and caps the count (`MAX_S2_IMAGES` / `MAX_S1_IMAGES`).
3. Computes the MNDWI median composite (or SAR fallback if S2 is empty).
4. Applies the water threshold → binary mask.
5. Exports a GeoTIFF thumbnail to the local `out_dir`.

**Expected run times (Google Colab Pro, Tesla V100):**

| Notebook | Approx. time |
|----------|-------------|
| Yearly (38 composites) | ~3–4 hours |
| Quarterly (44 composites) | ~1–2 hours |
| Bi-monthly (66 composites) | ~2–3 hours |

> The notebooks use thumbnail-based download (`getThumbURL` → `urllib`) rather than `Export.grid` tasks so they work interactively without needing the GEE batch task table.

### Step 6 — Verify and zip

After the loop finishes, a summary CSV and a bar chart are produced. The notebook then zips the entire `out_dir` into a single archive (e.g. `Yearly_WaterMasks_1988_2025.zip`) for easy download or transfer.

---

## 5. Output Convention

- **Format:** GeoTIFF, single band, 60 m resolution, EPSG:4326.
- **Pixel values:** `1 = water`, `0 = land`, `255` / `NaN` = no-data.
- **File naming:**
  - Yearly: `water_mask_YYYY.tif`
  - Quarterly: `water_mask_YYYY_QN.tif`  (N = 1–4)
  - Bi-monthly: `water_mask_YYYY_BMN.tif` (N = 1–6)

---

## 6. Environment Variables for Downstream Notebooks

After collection, point the preprocessing and model notebooks at your data by setting these environment variables (see [`docs/repository_guide.md`](docs/repository_guide.md) and the root [`README.md`](../README.md)):

```bash
set YEARLY_DIR=path/to/Yearly_WaterMasks_1988_2025
set QUARTERLY_DIR=path/to/Quarterly_WaterMasks_2015-2025
set BIMONTHLY_DIR=path/to/Sentinel_BiMonthly_WaterMasks_2015-2025
```

If the variables are not set, the notebooks fall back to `data/raw/yearly`, `data/raw/quarterly`, and `data/raw/bimonthly` respectively.

---

## 7. Relationship to the Rest of the Pipeline

```
data_collection/          preprocessing/           models/         analysis/
────────────────›  01_gap_filling_methods   ────────›  00_*_model_comparison
01_yearly_data_collection   02_long_term_prediction
02_bimonthly_data_collection
03_quarterly_data_collection
```

1. Run the three data collection notebooks → raw water-mask GeoTIFFs.
2. Run `preprocessing/01_gap_filling_methods.ipynb` → gap-filled masks + method comparison.
3. Proceed to the model training notebooks in `models/`.

See [`docs/repository_guide.md`](../docs/repository_guide.md) for the full end-to-end run order.

---

## 8. References

- Pre-defence report: [`docs/predefence_report.md`](../docs/predefence_report.md) — §4.2 *Data Acquisition*, §4.3 *Data Preprocessing*.
- Methodology summary: [`docs/methodology.md`](../docs/methodology.md).
- Root README: [`../README.md`](../README.md).