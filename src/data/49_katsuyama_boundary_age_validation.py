import pandas as pd
import numpy as np
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "44_katsuyama_original_52_full_records.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

doi = "10.1016/j.earscirev.2020.103173"
d = df[df["DOI"] == doi].copy()

d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
d["RelativeDepth"] = pd.to_numeric(d["RelativeDepth"], errors="coerce")

d = d.dropna(subset=["Age", "RelativeDepth"]).sort_values("RelativeDepth")

boundary_depth = -2.2
reference_age = 201.57
reference_uncertainty = 0.20
lower = reference_age - reference_uncertainty
upper = reference_age + reference_uncertainty

interpolated_age = np.interp(
    boundary_depth,
    d["RelativeDepth"],
    d["Age"]
)

nearest_index = (d["RelativeDepth"] - boundary_depth).abs().idxmin()
nearest = d.loc[nearest_index]

result = pd.DataFrame([{
    "boundary_depth_m": boundary_depth,
    "reference_age_Ma": reference_age,
    "reference_uncertainty_Ma": reference_uncertainty,
    "reference_lower_Ma": lower,
    "reference_upper_Ma": upper,
    "interpolated_2020_age_Ma": interpolated_age,
    "difference_from_reference_Ma": interpolated_age - reference_age,
    "inside_reference_interval": lower <= interpolated_age <= upper,
    "nearest_sample": nearest["SampleName"],
    "nearest_sample_depth_m": nearest["RelativeDepth"],
    "nearest_sample_age_Ma": nearest["Age"],
    "nearest_sample_age_inside_reference": (
        lower <= nearest["Age"] <= upper
    ),
    "number_of_2020_samples": len(d)
}])

print("=" * 72)
print("EXPERIMENT 49: EXTERNAL BOUNDARY AGE VALIDATION")
print("=" * 72)
print(result.T.to_string(header=False))
print("\n2020 samples surrounding the boundary:")
print(
    d.assign(
        distance_from_boundary=lambda x:
        (x["RelativeDepth"] - boundary_depth).abs()
    )
    .sort_values("distance_from_boundary")
    [["SampleName", "Age", "RelativeDepth", "Note"]]
    .head(6)
    .to_string(index=False)
)

output = base / "49_katsuyama_boundary_age_validation.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")

print("\nSaved:", output)
print("Original data unchanged.")
