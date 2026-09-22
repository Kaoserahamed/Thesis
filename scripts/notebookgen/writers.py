"""Notebook writers and the generator entry point."""

from __future__ import annotations

from .comparison import comparison_notebook
from .extract import ROOT, _banner, code, grab, md, write_notebook
from .metadata import ARCHITECTURES, ARCH_ORDER, FORECASTS, RESOLUTIONS, arch_number, stem_for
from .prediction import PRED_DATA, PRED_FORECAST, PRED_PRELUDE, PRED_VIZ
from .templates import (
    ARCH_CELL,
    LOAD_DATA,
    PRELUDE,
    SUMMARY_CELL,
    TRAIN_CELL,
    VIZ_CELL,
    architecture_code,
    callbacks_code,
    shared_pipeline_code,
)


def _title_markdown(res_key: str, arch_key: str) -> str:
    res, arch = RESOLUTIONS[res_key], ARCHITECTURES[arch_key]
    return (
        f"# {arch['label']} — {res['label']} Resolution\n\n"
        f"**Architecture {arch_number(arch_key)} of {len(ARCH_ORDER)}** "
        f"(`{stem_for(res_key, arch_key)}.ipynb`)\n\n"
        "**Stand-alone training notebook.** This file inlines the complete "
        "pipeline — configuration, preprocessing, losses/metrics, sequence "
        "generation, the strict temporal split, the `"
        + arch["label"]
        + "` architecture, training and evaluation — so it can "
        "be executed end-to-end without importing any project module.\n\n"
        f"- **Architecture:** `{arch['label']}` ({arch['section']}, "
        f"{arch['figure']}): {arch['summary']}\n"
        f"- **Temporal resolution:** {res['label']} ({res['period_name']} composite)\n"
        f"- **Sequence lengths L:** {res['sequence_lengths']}\n"
        "- **Temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025) and "
        "Setup 2 (train ≤ 2020 / test 2021–2025)\n"
        "- **Loss:** `L_BCE + L_Dice`; optimiser Adam(lr=1e-4, clipnorm=1.0)\n"
        "- **Evaluation:** IoU, Dice, Precision, Recall and signed area "
        "difference ΔA (km²)\n\n"
        f"> {res['best_note']}\n"
    )


def model_notebook(res_key: str, arch_key: str) -> None:
    """Write one self-contained architecture notebook."""
    res, arch = RESOLUTIONS[res_key], ARCHITECTURES[arch_key]
    lower = res["label"].lower()

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", lower),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%SEQLENS%%", repr(res["sequence_lengths"])),
            ("%%LABEL%%", arch["label"]),
            ("%%KEY%%", arch_key),
            ("%%BUILDER%%", arch["builder"]),
            ("%%SECTION%%", arch["section"]),
            ("%%FIGURE%%", arch["figure"]),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CODE%%", architecture_code(arch_key))
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        return t

    cells = [
        md(_title_markdown(res_key, arch_key)),
        md(
            "## 1. Configuration and imports\n\n"
            "Environment setup, imports, random seeds and the experiment "
            "configuration (thesis §5.9)."
        ),
        code(fill(PRELUDE)),
        md(
            "## 2. Preprocessing, losses, metrics and data pipeline\n\n"
            "These helpers are copied verbatim from the reviewed "
            "`utils/model_utils.py`, keeping this notebook fully self-contained: "
            "connected-component cleaning, water-mask loading, the BCE + Dice "
            "loss (equations 5.7–5.9), the IoU/Dice/area metrics (equations "
            "5.2–5.6), sliding-window sequences (§5.1) and the leakage-proof "
            "temporal split (§5.2)."
        ),
        code(shared_pipeline_code()),
        md(
            f"## 3. Load the {lower} water-mask time series\n\n"
            "Every GeoTIFF in the data directory is read, cleaned and resized "
            "to 256×256 binary masks."
        ),
        code(fill(LOAD_DATA)),
        md(
            f"## 4. Model architecture — {arch['label']}\n\n"
            f"Definition of the {arch['label']} network ({arch['section']}, "
            f"{arch['figure']}), inlined below."
        ),
        code(fill(ARCH_CELL)),
        md(
            "## 5. Training across sequence lengths and temporal setups\n\n"
            "The architecture is trained for every sequence length "
            f"{res['sequence_lengths']} under both temporal setups, with "
            "`ModelCheckpoint`, `EarlyStopping` and `ReduceLROnPlateau`."
        ),
        code(fill(TRAIN_CELL)),
        md("## 6. Comparative summary"),
        code(fill(SUMMARY_CELL)),
        md("## 7. Visualisations"),
        code(fill(VIZ_CELL)),
    ]
    path = ROOT / "models" / res_key / (stem_for(res_key, arch_key) + ".ipynb")
    write_notebook(path, cells)


def _prediction_title(res_key: str) -> str:
    res, fc = RESOLUTIONS[res_key], FORECASTS[res_key]
    best = ARCHITECTURES[res["best_key"]]
    extra = " and seasonal sin/cos channels" if fc["seasonal"] else ""
    return (
        f"# {fc['title']}\n\n"
        f"**Best model for the {res['label'].lower()} horizon** "
        f"(report {fc['report']}): `{best['label']}` ({best['section']}) with "
        f"sequence length L={fc['seq_len']}{extra}.\n\n"
        "This stand-alone notebook trains the winning architecture on the "
        "complete available record and rolls the prediction forward "
        f"autoregressively to produce the {fc['steps']}-step forecast "
        f"({fc['labels'][0]} … {fc['labels'][-1]}). It writes the forecast "
        "area table, the area-trend figure and the erosion/accretion risk "
        "map to `outputs/`.\n\n"
        f"> {res['best_note']}\n"
    )


def forecast_helpers_code() -> str:
    """Inline the pure-NumPy forecast-cube helpers from ``utils/forecast_utils.py``."""
    parts = [_banner("Forecast helpers  (utils/forecast_utils.py)")]
    for fn in ["frames_to_cube", "make_forecast_sequences"]:
        parts.append(grab(fn))
    return "\n\n".join(parts)


def final_prediction_notebook(res_key: str) -> None:
    """Write the best-model final-prediction notebook for one resolution."""
    res, fc = RESOLUTIONS[res_key], FORECASTS[res_key]
    lower = res["label"].lower()

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", lower),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%LABEL%%", ARCHITECTURES[res["best_key"]]["label"]),
            ("%%KEY%%", res["best_key"]),
            ("%%BUILDER%%", ARCHITECTURES[res["best_key"]]["builder"]),
            ("%%SEQLEN%%", str(fc["seq_len"])),
            ("%%TFC%%", str(fc["steps"])),
            ("%%FLABELS%%", repr(fc["labels"])),
            ("%%SEASONAL%%", "True" if fc["seasonal"] else "False"),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        t = t.replace("%%FORECAST_HELPERS%%", forecast_helpers_code())
        return t

    cells = [
        md(_prediction_title(res_key)),
        md(
            "## 1. Configuration and imports\n\n"
            "Environment setup, imports and the forecast configuration."
        ),
        code(fill(PRED_PRELUDE)),
        md(
            "## 2. Inlined pipeline (preprocessing, losses, metrics, split)\n\n"
            "The shared helpers, copied verbatim from `utils/model_utils.py`."
        ),
        code(shared_pipeline_code()),
        md(f"## 3. Load the full {lower} water-mask record"),
        code(fill(PRED_DATA)),
        md(
            f"## 4. Best model architecture\n\n"
            f"The {ARCHITECTURES[res['best_key']]['label']} network, inlined."
        ),
        code(architecture_code(res["best_key"])),
        md("## 5. Train on the full record and forecast autoregressively"),
        code(fill(PRED_FORECAST)),
        md("## 6. Forecast figures and risk map"),
        code(fill(PRED_VIZ)),
    ]
    path = ROOT / "analysis" / (fc["stem"] + ".ipynb")
    write_notebook(path, cells)


def main() -> None:
    print("Generating self-contained notebooks ...")
    for res_key in RESOLUTIONS:
        comparison_notebook(res_key)
    for res_key in RESOLUTIONS:
        for arch_key in ARCH_ORDER:
            model_notebook(res_key, arch_key)
    for res_key in FORECASTS:
        final_prediction_notebook(res_key)
    print("Done.")
