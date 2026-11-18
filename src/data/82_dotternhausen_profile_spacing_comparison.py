from pathlib import Path
import pandas as pd
import numpy as np

folder = Path("02_Data_Audit/provenance_analysis")
n_path = folder / "77_dotternhausen_original_source_table.csv"

nitrogen = pd.read_csv(n_path, encoding="utf-8-sig")
n_depths = sorted(nitrogen["Source_Depth_cm"].dropna().unique())

# Depth positions transcribed from Wang's dissertation,
# Table 2: stable carbon-isotope and related measurements.
carbon_depths = [
    613,621,624,628,631,635,638,642,645,648,652,656,
    670,673,677,680,684,687,691,694,698,701,705,708,
    712,715,719,722,726,729,733,736,740,743,747,750,
    754,757,761,764,768,775,778,781,785,788,796,801,
    805,815,817,822,825,828,831,834,836,840,843,848,
    850,853,856,859,862,869,871,876,883,894,898,899,
    903,905,914,917,919,921,922,926,935,940,949,956,
    961,964,971,976,979,981,986,993,1003,1012
]

carbon_depths = sorted(set(carbon_depths))
n_set = set(n_depths)
c_set = set(carbon_depths)

overlap = sorted(n_set & c_set)
n_only = sorted(n_set - c_set)
c_only = sorted(c_set - n_set)

def spacing_summary(values):
    values = np.array(sorted(set(values)), dtype=float)
    gaps = np.diff(values)
    return {
        "unique_depths": len(values),
        "minimum_depth_cm": values.min(),
        "maximum_depth_cm": values.max(),
        "depth_span_cm": values.max() - values.min(),
        "mean_spacing_cm": gaps.mean() if len(gaps) else np.nan,
        "median_spacing_cm": np.median(gaps) if len(gaps) else np.nan
    }

n_summary = spacing_summary(n_depths)
c_summary = spacing_summary(carbon_depths)

comparison = pd.DataFrame([
    {
        "profile": "2021 nitrogen isotope samples",
        **n_summary,
        "source": "Wang et al. (2021), Table 1 / Experiment 77"
    },
    {
        "profile": "Dissertation carbon isotope depth series",
        **c_summary,
        "source": "Wang dissertation (2021), Table 2"
    }
])

comparison.to_csv(
    folder / "82_dotternhausen_profile_spacing_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

overlap_df = pd.DataFrame({
    "nitrogen_sample_depth_cm": n_depths,
    "also_present_in_carbon_profile": [
        depth in c_set for depth in n_depths
    ]
})
overlap_df.to_csv(
    folder / "82_dotternhausen_depth_overlap.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 72)
print("EXPERIMENT 82: NITROGEN VS CARBON PROFILE SPACING")
print("=" * 72)

print("\nPROFILE COMPARISON")
print(comparison.to_string(index=False))

print("\nDEPTH OVERLAP")
print("Nitrogen sample depths:", len(n_depths))
print("Exact shared depth positions:", len(overlap))
print("Nitrogen-only positions:", len(n_only))
print("Carbon-profile-only positions:", len(c_only))
print("Nitrogen depth overlap percentage:",
      round(100 * len(overlap) / len(n_depths), 2))

print("\nNITROGEN DEPTHS ALSO IN CARBON PROFILE")
print(overlap)

print("\nIMPORTANT LIMITATIONS")
print("The carbon profile is a separate published geochemical series.")
print("Shared depth does not prove identical physical sample or aliquot.")
print("The carbon depth list was manually transcribed from the dissertation.")
print("No original isotope values, depths, or ages were modified.")

print("\nEXPERIMENT 82 COMPLETED")
