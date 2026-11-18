from pathlib import Path
import pandas as pd

folder = Path("02_Data_Audit/provenance_analysis")
source_path = folder / "78_dotternhausen_chronology_resolution_register.csv"

df = pd.read_csv(source_path, encoding="utf-8-sig")

def classify_depth(depth):
    if pd.isna(depth):
        return "unknown"
    if depth > 986 and depth <= 1050:
        return "bioturbated_organic_lean_unit_candidate"
    if depth == 986:
        return "boundary_ambiguous_check_original_figure"
    if depth >= 610 and depth < 986:
        return "laminated_organic_rich_unit_candidate"
    return "outside_documented_interval"

df["refined_lithological_class"] = (
    df["Source_Depth_cm"].apply(classify_depth)
)

df["classification_note"] = (
    "Depth-based provisional classification; not a sample-level "
    "lithology measurement unless independently documented."
)

output = folder / "79_dotternhausen_refined_lithology_register.csv"
df.to_csv(output, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 79: REFINED DOTTERNHAUSEN LITHOLOGY CLASSIFICATION")
print("=" * 72)

print("\nCLASSIFICATION COUNTS")
print(df["refined_lithological_class"].value_counts().to_string())

print("\nBOUNDARY RECORD")
print(
    df.loc[
        df["Source_Depth_cm"] == 986,
        ["Source_Depth_cm", "Source_d15Nbulk",
         "refined_lithological_class"]
    ].to_string(index=False)
)

print("\nNo raw values or age assignments modified.")
print("OUTPUT:", output)
print("EXPERIMENT 79 COMPLETED")
