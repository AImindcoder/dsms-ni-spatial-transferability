from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

base = Path("02_Data_Audit/provenance_analysis")

source = base / "56_independent_chronology_evidence_register.csv"
output = base / "57_katsuyama_2020_isotope_sequence.csv"
summary_file = base / "57_katsuyama_2020_isotope_sequence_summary.txt"

df = pd.read_csv(source, encoding="utf-8-sig")

required = [
    "SampleName",
    "SampleID_2020",
    "Age_2020",
    "Stage_2020",
    "Depth_2020",
    "Note_2020",
    "DSMSNI_δ15Nbulk_2020"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

result = df[required].copy()

for col in ["Age_2020", "Depth_2020", "DSMSNI_δ15Nbulk_2020"]:
    result[col] = pd.to_numeric(result[col], errors="coerce")

result["Bed_number"] = pd.to_numeric(
    result["Note_2020"].astype(str).str.extract(r"Bed\s*(\d+)", expand=False),
    errors="coerce"
)

result = result.sort_values(
    "Depth_2020",
    ascending=False,
    na_position="last"
).reset_index(drop=True)

result["isotope_missing"] = result["DSMSNI_δ15Nbulk_2020"].isna()

valid = result.dropna(
    subset=["Depth_2020", "Age_2020", "DSMSNI_δ15Nbulk_2020"]
).copy()

summary = [
    "EXPERIMENT 57 — KATSUYAMA 2020 ISOTOPE SEQUENCE AUDIT",
    "=" * 65,
    "",
    f"Total matched records: {len(result)}",
    f"Numeric isotope measurements: {result['DSMSNI_δ15Nbulk_2020'].notna().sum()}",
    f"Missing isotope measurements: {result['DSMSNI_δ15Nbulk_2020'].isna().sum()}",
    "",
    "ISOTOPE DISTRIBUTION",
    "-" * 40,
]

values = result["DSMSNI_δ15Nbulk_2020"].dropna()

if len(values):
    summary.extend([
        f"Mean: {values.mean():.5f}‰",
        f"Median: {values.median():.5f}‰",
        f"Standard deviation: {values.std():.5f}‰",
        f"Minimum: {values.min():.5f}‰",
        f"Maximum: {values.max():.5f}‰",
        f"Range: {values.max() - values.min():.5f}‰",
    ])

summary.extend([
    "",
    "MISSING MEASUREMENTS",
    "-" * 40,
])

for _, row in result[result["isotope_missing"]].iterrows():
    summary.append(
        f"{row['SampleName']} | {row['SampleID_2020']} | "
        f"Depth={row['Depth_2020']} | Bed={row['Note_2020']} | "
        f"Age={row['Age_2020']}"
    )

summary.extend([
    "",
    "DESCRIPTIVE STRATIGRAPHIC RELATIONSHIPS",
    "-" * 40,
])

if len(valid) >= 3:
    for label, x in [
        ("Depth versus isotope", valid["Depth_2020"]),
        ("Age versus isotope", valid["Age_2020"]),
    ]:
        rho, p = spearmanr(x, valid["DSMSNI_δ15Nbulk_2020"])
        tau, _ = kendalltau(x, valid["DSMSNI_δ15Nbulk_2020"])
        summary.append(
            f"{label}: Spearman rho={rho:.5f}; "
            f"Kendall tau={tau:.5f}; descriptive n={len(valid)}"
        )
else:
    summary.append("Insufficient complete observations.")

summary.extend([
    "",
    "SCIENTIFIC CAUTIONS",
    "-" * 40,
    "1. Values are DSMS-NI database values, not independently re-transcribed source values.",
    "2. Stratigraphically adjacent samples are not independent observations.",
    "3. Correlations are descriptive and must not be interpreted as independent statistical evidence.",
    "4. Missing isotope values remain missing.",
    "5. This experiment does not validate absolute geological ages.",
    "",
    "STRATIGRAPHICALLY ORDERED RECORDS",
    "-" * 40,
])

for _, row in result.iterrows():
    value = row["DSMSNI_δ15Nbulk_2020"]
    value_text = "MISSING" if pd.isna(value) else f"{value:.4f}‰"

    summary.append(
        f"{row['SampleName']} | "
        f"Depth={row['Depth_2020']} | "
        f"{row['Note_2020']} | "
        f"Age={row['Age_2020']} Ma | "
        f"{row['Stage_2020']} | "
        f"δ15N={value_text}"
    )

summary.extend([
    "",
    f"Output CSV: {output}",
])

result.to_csv(output, index=False, encoding="utf-8-sig")
summary_file.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nSaved:", output)
print("Saved:", summary_file)
