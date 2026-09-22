"""All-architecture comparison: inlined benchmark cells and their writer."""

from __future__ import annotations

from .extract import ROOT, code, md, write_notebook
from .metadata import RESOLUTIONS
from .templates import (
    PRELUDE,
    LOAD_DATA,
    all_architecture_code,
    callbacks_code,
    shared_pipeline_code,
)

COMP_TRAIN = """# ============================================================================
# Benchmark every architecture across setups and sequence lengths
# ============================================================================
%%CALLBACKS%%

rows = []
for arch_key, (arch_label, builder) in ARCH_REGISTRY.items():
    for setup_name, cfg in SETUP_CONFIGS.items():
        for seq_len in SEQUENCE_LENGTHS:
            tag = f"{arch_key}_L{seq_len}_{setup_name.replace(' ', '')}"
            print()
            print("=" * 78)
            print(f"  {arch_label} | {RESOLUTION} | {setup_name} | L={seq_len}")
            print("=" * 78)

            X_all, y_all, in_years, tgt_years = create_sequences(images, years, seq_len)
            X_tr, y_tr, X_val, y_val, X_test, y_test, ty_test = prepare_split(
                X_all, y_all, tgt_years, in_years, cfg["cutoff_year"])

            tf.keras.backend.clear_session()
            model = builder(seq_len)
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=DEFAULT_EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=create_callbacks(tag, checkpoint_dir=CKPT_DIR),
                      verbose=0)

            df = evaluate_model(model, X_test, y_test, ty_test,
                                PIXEL_AREA_KM2, arch_label, setup_name)
            df["L"] = seq_len
            rows.append(df)

df_all = pd.concat(rows, ignore_index=True)
df_summary = summarise_results(df_all)
print()
print("=== Mean metrics per architecture and setup ===")
df_summary
"""

COMP_VIZ = """# ============================================================================
# Architecture comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
df_summary.pivot(index="Model", columns="Setup", values="IoU").plot(
    kind="bar", ax=ax)
ax.set_ylabel("mean IoU")
ax.set_title(f"{RESOLUTION}: IoU by architecture and setup")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{RESOLUTION}_architecture_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Best architecture per setup (highest mean IoU)
best = (df_summary.sort_values("IoU", ascending=False)
        .groupby("Setup", as_index=False).first())
print("Best architecture per setup:")
best
"""


def comparison_notebook(res_key: str) -> None:
    """Write the self-contained all-architecture comparison notebook."""
    res = RESOLUTIONS[res_key]

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", res["label"].lower()),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%SEQLENS%%", repr(res["sequence_lengths"])),
            ("%%LABEL%%", "All architectures"),
            ("%%KEY%%", "comparison"),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        return t

    title = (
        f"# Full Model Comparison — {res['label']} Resolution\n\n"
        "**Stand-alone benchmark notebook.** Trains and evaluates **all five** "
        "spatiotemporal architectures — ConvLSTM, U-Net + LSTM, Attention "
        "U-Net + ConvLSTM, Swin ST and ViT — under identical preprocessing, "
        "sequence lengths, loss and strict temporal splitting, so the "
        "comparison is provably fair. Everything needed is inlined below.\n\n"
        f"- **Temporal resolution:** {res['label']} ({res['period_name']} composite)\n"
        f"- **Sequence lengths L:** {res['sequence_lengths']}\n"
        "- **Temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025) and "
        "Setup 2 (train ≤ 2020 / test 2021–2025)\n"
        "- **Loss:** `L_BCE + L_Dice`; optimiser Adam(lr=1e-4, clipnorm=1.0)\n\n"
        f"> {res['best_note']}\n"
    )

    cells = [
        md(title),
        md("## 1. Configuration and imports"),
        code(fill(PRELUDE)),
        md("## 2. Preprocessing, losses, metrics and data pipeline"),
        code(shared_pipeline_code()),
        md(f"## 3. Load the {res['label'].lower()} water-mask time series"),
        code(fill(LOAD_DATA)),
        md("## 4. All five architectures (inlined)"),
        code(all_architecture_code()),
        md("## 5. Benchmark every architecture"),
        code(fill(COMP_TRAIN)),
        md("## 6. Comparison figures"),
        code(fill(COMP_VIZ)),
    ]
    path = ROOT / "models" / ("00_" + res_key + "_model_comparison.ipynb")
    write_notebook(path, cells)
