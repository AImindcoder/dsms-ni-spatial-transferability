import pandas as pd
from pathlib import Path

base = Path(".")
source = base / "02_Data_Audit/provenance_analysis/41_cross_publication_sample_identity_candidates.csv"
out = base / "02_Data_Audit/provenance_analysis"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

doi_2016 = "10.1016/j.palaeo.2016.01.039"
doi_2020 = "10.1016/j.earscirev.2020.103173"

df = df[
    df["SiteName"].astype(str).str.strip().str.lower().eq("katsuyama")
    & df["DOI"].astype(str).str.strip().isin([doi_2016, doi_2020])
].copy()

target = next(c for c in df.columns if "15Nbulk" in c)
records = []

for sample, group in df.groupby("SampleName", dropna=False):
    old = group[group["DOI"].astype(str).str.strip() == doi_2016]
    new = group[group["DOI"].astype(str).str.strip() == doi_2020]

    if old.empty or new.empty:
        continue

    a = old.iloc[0]
    b = new.iloc[0]

    age16 = pd.to_numeric(pd.Series([a["Age"]]), errors="coerce").iloc[0]
    age20 = pd.to_numeric(pd.Series([b["Age"]]), errors="coerce").iloc[0]

    records.append({
        "SampleName": sample,
        "SampleID_2016": a["SampleID"],
        "SampleID_2020": b["SampleID"],
        "DOI_2016": a["DOI"],
        "DOI_2020": b["DOI"],
        "Age_2016_Ma": age16,
        "Age_2020_Ma": age20,
        "Absolute_Age_Difference_Ma": abs(age16-age20) if pd.notna(age16) and pd.notna(age20) else None,
        "Stage_2016": a["Stage"],
        "Stage_2020": b["Stage"],
        "Material_2016": a["Material"],
        "Material_2020": b["Material"],
        "Delta15Nbulk_2016": a[target],
        "Delta15Nbulk_2020": b[target],
        "Target_Missing_2016": pd.isna(a[target]),
        "Target_Missing_2020": pd.isna(b[target]),
        "RelativeDepth_2016": a["RelativeDepth"],
        "RelativeDepth_2020": b["RelativeDepth"],
        "Note_2016": a["Note"],
        "Note_2020": b["Note"],
        "Identity_Status": "cross_publication_sample_lineage_candidate_not_confirmed",
        "Chronology_Status": "requires_primary_source_verification",
        "Special_Flag": "IYF53D-0: investigate age assignment against published stratigraphy" if str(sample).strip() == "IYF53D-0" else "",
        "Interpretation": "Do not merge records or assume identical physical specimens without source-level evidence."
    })

result = pd.DataFrame(records).sort_values("SampleName")

csv_path = out / "42_katsuyama_cross_publication_lineage_register.csv"
txt_path = out / "42_katsuyama_cross_publication_lineage_summary.txt"

result.to_csv(csv_path, index=False, encoding="utf-8-sig")

lines = [
    "EXPERIMENT 42: KATSUYAMA CROSS-PUBLICATION LINEAGE REGISTER",
    "=" * 65,
    f"Candidate input rows: {len(df)}",
    f"Matched sample-name pairs: {len(result)}",
    f"2016 DOI: {doi_2016}",
    f"2020 DOI: {doi_2020}",
    f"2016 target missing: {int(result['Target_Missing_2016'].sum())}",
    f"2020 target missing: {int(result['Target_Missing_2020'].sum())}",
    f"Pairs with different assigned ages: {int((result['Absolute_Age_Difference_Ma'] > 0.01).sum())}",
    f"Median absolute age difference (Ma): {result['Absolute_Age_Difference_Ma'].median():.4f}",
    "",
    "IMPORTANT INTERPRETATION",
    "- These are lineage candidates, not confirmed duplicate physical samples.",
    "- No records were merged, deleted, or corrected.",
    "- The 2015 publication bed-number crosswalk uses a different numbering scheme",
    "  from some 2020 database Note values; direct bed-number matching is unsafe.",
    "- IYF53D-0 requires primary-source chronology verification.",
    "",
    f"Register: {csv_path}",
]

txt_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
