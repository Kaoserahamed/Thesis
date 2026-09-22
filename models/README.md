# Model Training — Spatiotemporal River Morphology Prediction

This folder contains the full suite of deep-learning experiments: **five spatiotemporal architectures** trained at three temporal resolutions (yearly, quarterly, bi-monthly), a head-to-head comparison notebook for each resolution, and best-model final-prediction notebooks.

> **Full methodology** is in the **pre-defence report**: [`../docs/predefence_report.md`](../docs/predefence_report.md) — see **Chapter 5** (§5.1–§5.9 *Methodology*) and **Chapter 7** (§7.1–§7.5 *Experimental Results*). Additional detail is in [`../docs/methodology.md`](../docs/methodology.md).

---

## Notebook design — self-contained, generated from one source

Every notebook in this folder is **stand-alone**: it inlines the complete
pipeline — imports and configuration, preprocessing, the BCE + Dice loss, the
IoU/Dice/area metrics, sliding-window sequence generation, the leakage-proof
temporal split, the model architecture, the training loop and the evaluation.
A reader can open any single notebook and run it end-to-end without importing a
project module.

To keep the notebooks from drifting apart, they are **generated** by
[`../scripts/generate_notebooks.py`](../scripts/generate_notebooks.py). That
script extracts the reviewed architecture and pipeline code from
[`../utils/model_utils.py`](../utils/model_utils.py) (via Python's `ast` module)
and inlines it, so `model_utils.py` remains the **single source of truth** while
the notebooks stay self-contained. Regenerate them with:

```bash
python scripts/generate_notebooks.py
```

---

## 1. Model Zoo

Five architectures are compared across all three resolutions. The registry key used in `build_model(name, seq_len)` is shown in parentheses.

| # | Architecture | Registry key | Strengths | Best resolution (IoU) |
|---|-------------|-------------|----------|----------------------|
| 1 | **ConvLSTM** | `convlstm` | Pure spatiotemporal recurrence; strong baseline | Quarterly (Setup 1) |
| 2 | **U-Net + LSTM** | `unet_lstm` | Encoder–decoder with ConvLSTM bottleneck; good at fine boundaries | **Bi-monthly (Setup 2): IoU = 0.7791** |
| 3 | **Attention U-Net + ConvLSTM** | `attention_unet_convlstm` | Attention-gated skip connections + ConvLSTM bottleneck; best hybrid | **Yearly: IoU = 0.7005** / Quarterly: IoU = 0.6956 |
| 4 | **Swin Spatio-Temporal Transformer** | `swin_st` | Hierarchical window attention + temporal fusion | Moderate performance |
| 5 | **Vision Transformer (ViT / ViViT)** | `vit_st` | Per-frame patch tokens + temporal transformer | Weakest (IoU ≈ 0.34–0.38) |

**Key finding:** Hybrid CNN-LSTM models (Architectures 1–3) consistently outperform pure transformers (4–5) for river morphology prediction.

---

## 2. Training Configuration

Every hyperparameter is defined inline in each notebook's *Configuration and imports* cell; the values mirror `utils/model_utils.py` (§0 *Global Defaults*) and thesis §5.9:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Framework** | TensorFlow 2.x (Keras) | |
| **Input shape** | `(seq_len, 256, 256, 1)` | 256×256 binary masks, single channel; 1 = water, 0 = land |
| **Mask convention** | `1 = water`, `0 = land`, `255`/`NaN` = no-data | Applied during `load_and_preprocess_image()` |
| **Optimiser** | `Adam(lr=1e-4, clipnorm=1.0)` | clipnorm prevents gradient explosion in ConvLSTM layers |
| **Loss** | `L_Total = L_BCE + L_Dice` | See §3 |
| **Metrics (Keras)** | `dice_coefficient`, `iou_metric` | Monitored during training |
| **Batch size** | 4 | Limited by 32 GB GPU memory |
| **Max epochs** | 200 | `DEFAULT_EPOCHS` |
| **EarlyStopping** | patience = 20, `restore_best_weights=True` | Monitors `val_loss` |
| **ReduceLROnPlateau** | patience = 7, factor = 0.5, `min_lr = 1e-7` | Halves LR when val_loss stalls |
| **ModelCheckpoint** | saves best model by `val_loss` | Saved to `outputs/checkpoints/` |
| **Random seed** | 42 (Python, NumPy, TensorFlow) | `seed_everything(42)` called before every build |
| **GPU** | memory growth enabled | `configure_gpu()` |
| **Mixed precision** | configurable (config.yaml) | `TF_CPP_MIN_LOG_LEVEL=3` to suppress XLA noise |

---

## 3. Loss Functions

### Combined loss (used for all five architectures)

```
L_Total = L_BCE + L_Dice
```

| Component | Formula | Role |
|-----------|---------|------|
| **L_BCE** (Binary Cross-Entropy) | `mean(BCE(y_true, y_pred))` | Stable pixel-level gradients, especially early in training |
| **L_Dice** (Dice loss) | `1 − Dice(y_true, y_pred)` | Region-level overlap; sharpens boundaries; compensates for class imbalance |

The sum balances pixel-wise accuracy (BCE) with region-level overlap (Dice), critical for water-body segmentation where water pixels are a minority of the 256×256 frame.

```python
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)   # Eq. 5.7

def bce_loss(y_true, y_pred):
    return K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))   # Eq. 5.8
```

---

## 4. Evaluation Metrics

### Keras metrics (training time)

| Metric | Function | Behaviour |
|--------|----------|-----------|
| **Dice coefficient** | `dice_coefficient(y_true, y_pred)` | Soft, differentiable; used as a training metric |
| **IoU** | `iou_metric(y_true, y_pred)` | Binarised IoU at threshold 0.5; used as a training metric |

### NumPy evaluation metrics (post-training, test set)

All computed by `evaluate_model()` → `summarise_results()`:

| Metric | Formula | Unit |
|--------|---------|------|
| **IoU** | `(Intersection + ε) / (Union + ε)` | — |
| **Dice** | `2·TP / (2·TP + FP + FN)` | — |
| **Precision** | `TP / (TP + FP)` | — |
| **Recall** | `TP / (TP + FN)` | — |
| **Area difference (ΔA)** | `A_pred − A_true` | km² |

Threshold for binarisation: **0.5**. `evaluate_model()` also computes a **persistence baseline** (last input frame as prediction) for every sample.

---

## 5. Sequence Generation

Sequences are built with a **sliding window** (stride = 1, horizon = 1) from `create_sequences()`:

```
Input:  X_i = {X_i, ..., X_{i+L-1}}   (L consecutive frames)
Output: y_i = X_{i+L}                   (next frame)
```

| Resolution | Sequence length options (L) | Stride | Horizon |
|------------|-----------------------------|--------|---------|
| **Yearly** | `{4, 5, 6}` years | 1 year | 1 year |
| **Quarterly** | `{6, 8, 10}` quarters | 1 quarter | 1 quarter |
| **Bi-monthly** | `{6, 9, 12}` bi-months | 1 bi-month | 1 bi-month |

Multiple L values are tested per resolution. Overlapping windows (stride = 1) maximise sample count.

---

## 6. Strict Temporal Split

Two experimental setups (thesis §5.2), implemented by `prepare_split()`:

| Setup | Train+Val (target ≤ cutoff) | Test (target > cutoff) | % train |
|-------|----------------------------|------------------------|---------|
| **Setup 1** | up to 2015 | 2016–2025 | 85% / 15% |
| **Setup 2** | up to 2020 | 2021–2025 | 85% / 15% |

`prepare_split()` asserts **no year overlap** between train/val and test. Best results are reported under **Setup 1** (larger training set); Setup 2 is used for the short-term bi-monthly forecast.

---

## 7. Notebook Catalogue

### 7.1 Yearly Models

| # | Notebook | Architecture | Notes |
|---|----------|--------------|-------|
| — | `00_yearly_model_comparison.ipynb` | **All 5** | Self-contained full comparison at L ∈ {4, 5, 6} |
| 1 | `yearly/01_yearly_convlstm.ipynb` | ConvLSTM | Self-contained |
| 2 | `yearly/02_yearly_unet_lstm.ipynb` | U-Net + LSTM | Self-contained |
| 3 | `yearly/03_yearly_attention_unet_convlstm.ipynb` | Attention U-Net + ConvLSTM | **Best yearly model** (IoU = 0.7005, L=5, Setup 1) |
| 4 | `yearly/04_yearly_swin_transformer.ipynb` | Swin ST | Self-contained |
| 5 | `yearly/05_yearly_vit.ipynb` | ViT ST | Self-contained |

**Final prediction (best model):** [`analysis/02_long_term_prediction.ipynb`](../analysis/02_long_term_prediction.ipynb) — 2026–2040 yearly forecast.

### 7.2 Quarterly Models

| # | Notebook | Architecture | Notes |
|---|----------|--------------|-------|
| — | `00_quarterly_model_comparison.ipynb` | **All 5** | Self-contained full comparison at L ∈ {6, 8, 10} |
| 1 | `quarterly/01_quarterly_convlstm.ipynb` | ConvLSTM | Self-contained |
| 2 | `quarterly/02_quarterly_unet_lstm.ipynb` | U-Net + LSTM | Self-contained |
| 3 | `quarterly/03_quarterly_attention_unet_convlstm.ipynb` | Attention U-Net + ConvLSTM | **Best quarterly model** (IoU = 0.6956, L=8, Setup 1) |
| 4 | `quarterly/04_quarterly_swin_transformer.ipynb` | Swin ST | Self-contained |
| 5 | `quarterly/05_quarterly_vit.ipynb` | ViViT | Self-contained |

**Final prediction (best model):** [`analysis/04_quarterly_prediction.ipynb`](../analysis/04_quarterly_prediction.ipynb) — 2026 Q1–2029 Q4 quarterly forecast.

### 7.3 Bi-monthly Models

| # | Notebook | Architecture | Notes |
|---|----------|--------------|-------|
| — | `00_bimonthly_model_comparison.ipynb` | **All 5** | Self-contained full comparison at L ∈ {6, 9, 12} |
| 1 | `bimonthly/01_bimonthly_convlstm.ipynb` | ConvLSTM | Self-contained |
| 2 | `bimonthly/02_bimonthly_unet_lstm.ipynb` | U-Net + LSTM | **Best bi-monthly model** (IoU = 0.7791, Setup 2, L=9) |
| 3 | `bimonthly/03_bimonthly_attention_unet_convlstm.ipynb` | Attention U-Net + ConvLSTM | Self-contained |
| 4 | `bimonthly/04_bimonthly_swin_transformer.ipynb` | Swin ST | Self-contained |
| 5 | `bimonthly/05_bimonthly_vit.ipynb` | ViT ST | Self-contained |

> **Numbering convention.** All three resolutions use the same numbering, so the
> five notebooks in each folder line up one-to-one and can be compared directly:
> **`01` ConvLSTM · `02` U-Net + LSTM · `03` Attention U-Net + ConvLSTM ·
> `04` Swin ST · `05` ViT.** The `00_*` comparison notebook at the folder root
> benchmarks all five at once.

**Final prediction (best model):** [`analysis/03_short_term_prediction.ipynb`](../analysis/03_short_term_prediction.ipynb) — 2026 P1–2028 P6 bi-monthly forecast.

---

## 8. Architecture Details

### Architecture 1 — ConvLSTM  (`convlstm`)

- **Input:** `(L, 256, 256, 1)`
- Encoder: TimeDistributed Conv2D(32)+BN ×2 + MaxPool2D + Dropout(0.2) → Conv2D(64)+BN ×2 + MaxPool2D + Dropout(0.2)
- Bottleneck: **ConvLSTM2D(128, 3×3, return_sequences=True, dropout=0.3, L2=1e-3)** → BN → **ConvLSTM2D(64, 3×3, return_sequences=False, dropout=0.3, L2=1e-3)** → BN
- Decoder: Upsampling2D → Conv2D(64)+BN+Dropout → Upsampling2D → Conv2D(32)+BN+Dropout → Conv2D(16) → Conv2D(1, sigmoid)

### Architecture 2 — U-Net + LSTM  (`unet_lstm`)

- Encoder: 3 levels (Conv2D 32→64→128, each ×2 + BN + MaxPool2D + Dropout 0.15/0.2/0.2)
- Bottleneck: Conv2D(256)+BN → **ConvLSTM2D(256, return_sequences=True, dropout=0.3)** → **ConvLSTM2D(128, return_sequences=False, dropout=0.3)**
- Decoder: 3 upsampling levels with **skip connections** from encoder
- Final: Conv2D(16) → Conv2D(1, sigmoid)

### Architecture 3 — Attention U-Net + ConvLSTM  (`attention_unet_convlstm`)

- Same encoder/decoder as U-Net+LSTM
- **AttentionGate** (Oktay et al., 2018) on each skip connection: gating signal highlights relevant regions before concatenation
- Same ConvLSTM bottleneck
- **Best hybrid model** — attention gates improve boundary precision

### Architecture 4 — Swin Spatio-Temporal Transformer  (`swin_st`)

- Shared spatial encoder per frame: `patch_embed` (Conv2D 64, k=4, s=4) → `window_attention` (Conv2D+MultiHeadAttention, 2/4/8 heads) × 3 stages
- **Temporal fusion:** Concatenate encoded frames → Conv1D(256) + MultiHeadAttention(4 heads) + GlobalAveragePooling1D → Dense(16×16×256) → Reshape
- U-Net decoder with skip connections

### Architecture 5 — ViT / ViViT  (`vit_st`)

- Spatial encoder per frame: PatchEmbed(Conv2D embed_dim=128, patch_size=16) → LayerNorm → MultiHeadAttention(4 heads) → LayerNorm → GlobalAveragePooling1D

### Architecture 4 — Swin Spatio-Temporal Transformer  (`swin_st`)

- Shared spatial encoder per frame: `patch_embed` (Conv2D 64, k=4, s=4) → `window_attention` (Conv2D+MultiHeadAttention, 2/4/8 heads) × 3 stages
- **Temporal fusion:** Concatenate encoded frames → Conv1D(256) + MultiHeadAttention(4 heads) + GlobalAveragePooling1D → Dense(16×16×256) → Reshape
- U-Net decoder with skip connections

### Architecture 5 — ViT / ViViT  (`vit_st`)

- Spatial encoder per frame: PatchEmbed(Conv2D embed_dim=128, patch_size=16) → LayerNorm → MultiHeadAttention(4 heads) → LayerNorm → GlobalAveragePooling1D
- Temporal transformer: MultiHeadAttention on frame tokens → LayerNorm → Dense(256, gelu) → GAP → Dense(16×16×128) → Reshape
- Conv2DTranspose decoder (4 stages) → Conv2D(1, sigmoid)

---

## 9. Results Summary

### Best architecture per resolution

| Resolution | Best model | Setup | Seq. length | IoU | Dice |
|------------|-----------|-------|-------------|-----|------|
| **Yearly** | Attention U-Net+ConvLSTM | Setup 1 (10-yr) | 5 | **0.7005** | **0.8236** |
| **Quarterly** | Attention U-Net+ConvLSTM | Setup 1 (long-range) | — | **0.6956** | 0.8139 |
| **Bi-monthly** | U-Net+LSTM | Setup 2 (medium-range) | 9 | **0.7791** | **0.8701** |

### All architectures — yearly (Setup 1, representative)

| Architecture | IoU | Dice |
|-------------|-----|------|
| ConvLSTM | ~0.65–0.68 | ~0.78–0.81 |
| U-Net+LSTM | ~0.67–0.69 | ~0.80–0.82 |
| Attention U-Net+ConvLSTM | **0.7005** | **0.8236** |
| Swin ST | ~0.45–0.55 | ~0.60–0.70 |
| ViT ST | ~0.34–0.38 | ~0.50–0.55 |

> ViT/ViViT reached only IoU ≈ 0.34–0.38, confirming that pure transformers underperform. Hybrid CNN-LSTM models are consistently superior.

---

## 10. Environment Variables

| Variable | Default | Required? | Description |
|----------|---------|-----------|-------------|
| `YEARLY_DIR` | `data/raw/yearly` | Yes (yearly) | Path to yearly water-mask GeoTIFFs |
| `QUARTERLY_DIR` | `data/raw/quarterly` | Yes (quarterly) | Path to quarterly water-mask GeoTIFFs |
| `BIMONTHLY_DIR` | `data/raw/bimonthly` | Yes (bi-monthly) | Path to bi-monthly water-mask GeoTIFFs |

Outputs: `outputs/predictions/` (forecast masks + figures) and `outputs/checkpoints/` (best model weights).

---

## 11. Model Comparison Notebooks

Each `00_*_model_comparison.ipynb` does the following:

1. **Loads** the image stack from `DATA_DIRS[resolution]`.

---

## 12. Final Prediction Notebooks

Each prediction notebook trains the **winning architecture** on the whole
available record and rolls the forecast forward autoregressively, writing the
forecast-area CSV plus area-trend, risk-map and sample figures to
`outputs/<resolution>/forecast/`.

### [`analysis/02_long_term_prediction.ipynb`](../analysis/02_long_term_prediction.ipynb) — Yearly, 2026–2040

- Best yearly model: **Attention U-Net + ConvLSTM, L=5, Setup 1** (report §7.1).
- 15-step autoregressive roll-out (each predicted year is fed back as input).
- Produces the forecast water-area series (2026–2040) and the water-occurrence
  **risk map** (report Figure 7.1).
- Key result: **≈ 290 km² expansion (+56%)** over the 2025 baseline, dominated
  by accretion.

### [`analysis/04_quarterly_prediction.ipynb`](../analysis/04_quarterly_prediction.ipynb) — Quarterly, 2026 Q1–2029 Q4

- Best quarterly model: **Attention U-Net + ConvLSTM, L=8, Setup 1** (report
  §7.2, Tables 7.10 and 7.13).
- 16-step quarterly roll-out with area trend and risk map.

### [`analysis/03_short_term_prediction.ipynb`](../analysis/03_short_term_prediction.ipynb) — Bi-monthly, 2026 P1–2028 P6

- Best bi-monthly model: **U-Net + LSTM, L=9, Setup 2** (report §7.3).
- 18-period forecast with **seasonal channels** (sin/cos encoding of the
  intra-annual position).
- Reproduces the §7.5 short-term bi-monthly forecast.

---

## 13. File Organisation

```
models/
├── 00_yearly_model_comparison.ipynb     # All 5 architectures, self-contained
├── 00_quarterly_model_comparison.ipynb
├── 00_bimonthly_model_comparison.ipynb
│
├── yearly/                              # 5 architectures @ yearly resolution
│   ├── 01_yearly_convlstm.ipynb
│   ├── 02_yearly_unet_lstm.ipynb
│   ├── 03_yearly_attention_unet_convlstm.ipynb     # best yearly architecture
│   ├── 04_yearly_swin_transformer.ipynb
│   └── 05_yearly_vit.ipynb
│
├── quarterly/                           # 5 architectures @ quarterly resolution
│   ├── 01_quarterly_convlstm.ipynb
│   ├── 02_quarterly_unet_lstm.ipynb
│   ├── 03_quarterly_attention_unet_convlstm.ipynb  # best quarterly architecture
│   ├── 04_quarterly_swin_transformer.ipynb
│   └── 05_quarterly_vit.ipynb
│
└── bimonthly/                           # 5 architectures @ bi-monthly resolution
    ├── 01_bimonthly_convlstm.ipynb
    ├── 02_bimonthly_unet_lstm.ipynb                 # best bi-monthly architecture
    ├── 03_bimonthly_attention_unet_convlstm.ipynb
    ├── 04_bimonthly_swin_transformer.ipynb
    └── 05_bimonthly_vit.ipynb

analysis/
├── 01_gap_filling_comparison.ipynb  # Gap-filling benchmark (Chapter 4)
├── 02_long_term_prediction.ipynb    # Yearly 2026-2040 forecast + risk map
├── 03_short_term_prediction.ipynb   # Bi-monthly 2026 P1-2028 P6 forecast
└── 04_quarterly_prediction.ipynb    # Quarterly 2026 Q1-2029 Q4 forecast

outputs/
└── <resolution>/
    ├── checkpoints/   # Best model weights (.keras)
    └── forecast/      # Forecast CSV + figures (prediction notebooks)
```

---

## 14. References

- Pre-defence report: [`docs/predefence_report.md`](docs/predefence_report.md) — **Chapter 5** (§5.1–§5.9) and **Chapter 7** (§7.1–§7.5).
- Methodology: [`docs/methodology.md`](docs/methodology.md).
- Repository guide: [`docs/repository_guide.md`](docs/repository_guide.md).
- Root README: [`../README.md`](../README.md).
- Shared library (single source of truth): [`utils/model_utils.py`](../utils/model_utils.py) — all architectures, losses, metrics, sequences, splits, callbacks, evaluation; the notebooks are generated from it.
- Notebook generator: [`scripts/generate_notebooks.py`](../scripts/generate_notebooks.py) — inlines `model_utils.py` into the self-contained notebooks.
- Visualisation: [`utils/visualization_utils.py`](../utils/visualization_utils.py) — `plot_training_curves()`, `plot_sequence_comparison()`, `plot_predictions()`, `plot_area_trend()`, `compute_change_frequencies()`, `plot_risk_maps()`, `create_folium_risk_map()`.