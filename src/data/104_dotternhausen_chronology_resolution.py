from pathlib import Path
import pandas as pd

root = Path(".")
source = pd.read_csv(
    root / "01_Data/interim/100_dotternhausen_verified_source_crosswalk.csv"
)

depth = pd.to_numeric(source["Depth_cm"], errors="coerce")
age = pd.to_numeric(source["Age"], errors="coerce")

audit = source[["SampleID", "Depth_cm", "Age"]].copy()
audit["source_stratigraphic_interval"] = (
    "upper semicaletum subzone to upper elegans subzone"
)
audit["database_age_resolution"] = "one repeated numerical age"
audit["chronology_interpretation"] = (
    "broad Toarcian age assignment; not sample-specific age control"
)
audit["chronology_caution"] = (
    "do not interpret depth variation as calibrated time variation"
)

out = root / "01_Data/interim/104_dotternhausen_chronology_resolution.csv"
audit.to_csv(out, index=False, encoding="utf-8-sig")

report = f"""
DOTTERNHAUSEN CHRONOLOGICAL RESOLUTION AUDIT
{'=' * 65}

Records: {len(source)}
Unique numerical ages: {age.nunique()}
Assigned age: {age.dropna().iloc[0]:.5f} Ma
Unique stratigraphic depths: {depth.nunique()}
Depth range: {depth.min():.1f}–{depth.max():.1f} cm
Depth span: {depth.max() - depth.min():.1f} cm

SOURCE STRATIGRAPHIC CONTEXT
Upper semicaletum subzone to upper elegans subzone.
Source: Wang et al. (2021), DOI 10.1029/2020PA004207.

DATABASE REPRESENTATION
All records have one repeated numerical age.
The database representation does not preserve sample-specific
numerical ages across the source's stratigraphic subdivisions.

INTERPRETATION
This is evidence of reduced chronological resolution in the
database representation. It is not proof that the assigned
age is incorrect. The value may be a broad geological-stage
representative age rather than a sample-specific age.

RESEARCH IMPLICATION
Do not treat the repeated Age value as independent temporal
information for these 31 observations. Retain depth and source
stratigraphic context as separate fields.

Output: {out}
"""
report_path = root / "01_Data/interim/104_dotternhausen_chronology_resolution_audit.txt"
report_path.write_text(report.strip() + "\n", encoding="utf-8")

print(report)
print("Saved chronology-resolution audit.")
