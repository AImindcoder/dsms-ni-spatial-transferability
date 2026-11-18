from pathlib import Path
import pandas as pd
import numpy as np

folder = Path("02_Data_Audit/provenance_analysis")
source = folder / "79_dotternhausen_refined_lithology_register.csv"

df = pd.read_csv(source, encoding="utf-8-sig")
depths = np.sort(df["Source_Depth_cm"].dropna().to_numpy())

n = len(depths)
span = float(depths[-1] - depths[0])
gaps = np.diff(depths)

published_spacing = 3.5
observed_spacing = float(gaps.mean())
expected_intervals = span / published_spacing
expected_samples = expected_intervals + 1

records = [
    {
        "evidence_item": "published_sampling_interval",
        "value": "1050–610 cm",
        "evidence_source": "Wang et al. (2021), Section 3.1",
        "interpretation": "Investigated stratigraphic interval"
    },
    {
        "evidence_item": "published_average_spacing_cm",
        "value": published_spacing,
        "evidence_source": "Wang et al. (2021), Section 3.1",
        "interpretation": "Reported average sampling spacing"
    },
    {
        "evidence_item": "published_sample_count",
        "value": 31,
        "evidence_source": "Wang et al. (2021), Section 3.1",
        "interpretation": "Reported samples used for geochemical analyses"
    },
    {
        "evidence_item": "table_1_sample_count",
        "value": n,
        "evidence_source": "Wang et al. (2021), Table 1",
        "interpretation": "Number of depth records independently extracted"
    },
    {
        "evidence_item": "table_1_depth_span_cm",
        "value": span,
        "evidence_source": "Wang et al. (2021), Table 1",
        "interpretation": "Maximum minus minimum tabulated depth"
    },
    {
        "evidence_item": "calculated_mean_adjacent_spacing_cm",
        "value": observed_spacing,
        "evidence_source": "Experiment 80, calculated from Table 1",
        "interpretation": "Mean of 30 adjacent depth intervals"
    },
    {
        "evidence_item": "calculated_median_adjacent_spacing_cm",
        "value": float(np.median(gaps)),
        "evidence_source": "Experiment 80, calculated from Table 1",
        "interpretation": "Median of adjacent depth intervals"
    },
    {
        "evidence_item": "expected_intervals_at_published_spacing",
        "value": expected_intervals,
        "evidence_source": "Independent arithmetic",
        "interpretation": "Depth span divided by reported average spacing"
    },
    {
        "evidence_item": "approx_expected_samples_at_published_spacing",
        "value": round(expected_samples),
        "evidence_source": "Independent arithmetic",
        "interpretation": "Approximate sample count needed across the tabulated span"
    },
    {
        "evidence_item": "observed_to_published_spacing_ratio",
        "value": observed_spacing / published_spacing,
        "evidence_source": "Independent arithmetic",
        "interpretation": "Calculated mean divided by published mean"
    },
    {
        "evidence_item": "review_status",
        "value": "apparent_internal_sampling_description_inconsistency",
        "evidence_source": "Combined publication and table audit",
        "interpretation": "Requires clarification; no published value corrected"
    }
]

out = pd.DataFrame(records)
output = folder / "81_dotternhausen_sampling_description_evidence_register.csv"
out.to_csv(output, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 81: PUBLISHED SAMPLING DESCRIPTION CONSISTENCY")
print("=" * 72)
print(out.to_string(index=False))
print("\nIMPORTANT:")
print("No original sample depths, isotope values, or age assignments changed.")
print("The 3.5 cm published figure is preserved as reported.")
print("The discrepancy remains unresolved pending source clarification.")
print("\nOUTPUT:", output)
print("EXPERIMENT 81 COMPLETED")
