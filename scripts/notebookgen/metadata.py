"""Experiment metadata: resolutions, architectures, setups and forecasts.

Mirrors ``docs/predefence_report.md`` (sections 5.1 and 7); every notebook the
generator writes is derived from this single description.
"""

from __future__ import annotations

from typing import Any, Dict

RESOLUTIONS: Dict[str, Dict[str, Any]] = {
    "yearly": {
        "label": "Yearly",
        "sequence_lengths": [4, 5, 6],
        "period_name": "year",
        "env_var": "YEARLY_DIR",
        "best_key": "attention_unet_convlstm",
        "best_note": (
            "Best architecture for the yearly horizon (report §7.1): "
            "**Attention U-Net + ConvLSTM**, Setup 1, L=5 -- "
            "IoU = 0.7005, Dice = 0.8236."
        ),
    },
    "quarterly": {
        "label": "Quarterly",
        "sequence_lengths": [6, 8, 10],
        "period_name": "quarter",
        "env_var": "QUARTERLY_DIR",
        "best_key": "attention_unet_convlstm",
        "best_note": (
            "Best architecture for the quarterly horizon (report §7.2): "
            "**Attention U-Net + ConvLSTM**, Setup 1 -- "
            "IoU = 0.6956, Dice = 0.8139."
        ),
    },
    "bimonthly": {
        "label": "Bi-monthly",
        "sequence_lengths": [6, 9, 12],
        "period_name": "bi-month",
        "env_var": "BIMONTHLY_DIR",
        "best_key": "unet_lstm",
        "best_note": (
            "Best architecture for the bi-monthly horizon (report §7.3): "
            "**U-Net + LSTM**, Setup 2, L=9 -- "
            "IoU = 0.7791, Dice = 0.8701."
        ),
    },
}

# Every architecture, its Keras builder, the helper symbols that must travel
# with it, the report section that specifies it, and the delivery file stem.
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "convlstm": {
        "label": "ConvLSTM",
        "builder": "build_convlstm",
        "helpers": [],
        "section": "§5.5",
        "figure": "Figure 5.1",
        "summary": (
            "A standalone spatiotemporal recurrent network: per-frame "
            "convolutional feature extraction followed by stacked "
            "ConvLSTM2D layers that model temporal evolution and a "
            "lightweight decoder that restores the full resolution."
        ),
        "short": "convlstm",
    },
    "unet_lstm": {
        "label": "U-Net + LSTM",
        "builder": "build_unet_lstm",
        "helpers": ["_conv_bn"],
        "section": "§5.6",
        "figure": "Figure 5.2",
        "summary": (
            "An encoder-decoder network whose encoder is applied per "
            "frame and whose ConvLSTM bottleneck captures temporal "
            "dependencies; skip connections preserve fine river "
            "boundaries. Best model for the bi-monthly horizon."
        ),
        "short": "unet_lstm",
    },
    "attention_unet_convlstm": {
        "label": "Attention U-Net + ConvLSTM",
        "builder": "build_attention_unet_convlstm",
        "helpers": ["_conv_bn", "AttentionGate"],
        "section": "§5.7",
        "figure": "Figure 5.3",
        "summary": (
            "A U-Net with a ConvLSTM bottleneck and additive "
            "attention-gated skip connections (Oktay et al., 2018) "
            "that focus the decoder on the spatially relevant regions. "
            "Best model for the yearly and quarterly horizons."
        ),
        "short": "attention_unet_convlstm",
    },
    "swin_st": {
        "label": "Swin Spatio-Temporal Transformer",
        "builder": "build_swin_st",
        "helpers": ["_conv_bn"],
        "section": "§5.8",
        "figure": "Figure 5.4",
        "summary": (
            "A hierarchical window-attention encoder shared across "
            "frames, a temporal-fusion head and a U-Net-style "
            "convolutional decoder."
        ),
        "short": "swin_transformer",
    },
    "vit_st": {
        "label": "Vision Transformer (ViViT)",
        "builder": "build_vit_st",
        "helpers": ["_conv_bn"],
        "section": "§5.9",
        "figure": "Figure 5.5",
        "summary": (
            "Per-frame patch tokenisation with a shared spatial "
            "transformer, a temporal transformer across frames, and a "
            "convolutional decoder that reconstructs the mask."
        ),
        "short": "vit",
    },
}

# Canonical presentation order of the five architectures.  Notebooks are numbered
# 01..05 in exactly this order, identically for all three resolutions, so the
# yearly/, quarterly/ and bimonthly/ folders stay parallel:
#
#     01_convlstm  02_unet_lstm  03_attention_unet_convlstm
#     04_swin_transformer  05_vit
ARCH_ORDER: list = [
    "convlstm",
    "unet_lstm",
    "attention_unet_convlstm",
    "swin_st",
    "vit_st",
]
assert list(ARCHITECTURES) == ARCH_ORDER, "ARCHITECTURES must follow ARCH_ORDER"


def arch_number(arch_key: str) -> int:
    """1-based position of an architecture in the canonical order (1..5)."""
    return ARCH_ORDER.index(arch_key) + 1


def stem_for(res_key: str, arch_key: str) -> str:
    """Notebook file stem, e.g. ``03_yearly_attention_unet_convlstm``."""
    return f"{arch_number(arch_key):02d}_{res_key}_{ARCHITECTURES[arch_key]['short']}"


SETUP_CONFIGS = [
    ("Setup 1", 2015, "2016-2025"),
    ("Setup 2", 2020, "2021-2025"),
]


FORECASTS: Dict[str, Dict[str, Any]] = {
    "yearly": {
        "stem": "02_long_term_prediction",
        "seq_len": 5,
        "seasonal": False,
        "steps": 15,
        "labels": [str(y) for y in range(2026, 2041)],
        "title": "Long-Term Morphological Forecast (2026–2040)",
        "report": "section 7.4",
    },
    "quarterly": {
        "stem": "04_quarterly_prediction",
        "seq_len": 8,
        "seasonal": False,
        "steps": 16,
        "labels": [str(2026 + i // 4) + " Q" + str(i % 4 + 1) for i in range(16)],
        "title": "Quarterly Morphological Forecast (2026 Q1–2029 Q4)",
        "report": "section 7.2",
    },
    "bimonthly": {
        "stem": "03_short_term_prediction",
        "seq_len": 9,
        "seasonal": True,
        "steps": 18,
        "labels": [str(2026 + i // 6) + " P" + str(i % 6 + 1) for i in range(18)],
        "title": "Short-Term Morphological Forecast (2026 P1–2028 P6)",
        "report": "section 7.5",
    },
}
