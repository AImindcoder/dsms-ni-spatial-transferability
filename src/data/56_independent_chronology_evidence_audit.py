from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
BASE = ROOT / "02_Data_Audit" / "provenance_analysis"

crosswalk_file = BASE / "53b_katsuyama_corrected_2015_bed_crosswalk.csv"
original_file = BASE / "44_katsuyama_original_52_full_records.csv"

output_file = BASE / "56_independent_chronology_evidence_register.csv"
summary_file = BASE / "56_independent_chronology_evidence_summary.txt"

crosswalk = pd.read_csv(crosswalk_file, encoding="utf-8-sig")
original = pd.read_csv(original_file, encoding="utf-8-sig")

if len(crosswalk) != 26:
    raise ValueError(f"Expected 26 pairs in 53b; found {len(crosswalk)}")

if "SampleID" not in original.columns:
    raise ValueError("SampleID column not found in original-record export.")

target_col = next(
    (c for c in original.columns if "15Nbulk" in str(c)),
    None
)
tech_col = next(
    (c for c in original.columns if str(c) == "Techniqueδ15N"),
    None
)

if target_col is None:
    raise ValueError("Could not identify the original δ15Nbulk column.")

if tech_col is None:
    raise ValueError("Could not identify the Techniqueδ15N column.")

original = original.set_index("SampleID", drop=False)

# Independently documented sample-to-bed crosswalk from
# Fujisaki et al. (2015), Table 3.
bed_evidence = {
    "IYF53E-1": (8, "Jurassic"),
    "IYF53E-0": (9, "Jurassic"),
    "IYF53D-3": (11, "Jurassic"),
    "IYF53D-2": (12, "Jurassic"),
    "IYF53D-1": (13, "Jurassic"),
    "IYF53D-0": (14, "Boundary"),
    "IYF53B-0": (15, "Triassic"),
}

def period_from_stage(stage):
    if pd.isna(stage):
        return "Unresolved"

    s = str(stage).strip().lower()

    triassic = [
        "induan", "olenekian", "anisian",
        "ladinian", "carnian", "norian", "rhaetian"
    ]
    jurassic = [
        "hettangian", "sinemurian", "pliensbachian",
        "toarcian", "aalenian", "bajocian",
        "bathonia", "callovian", "oxfordian",
        "kimmeridgian", "tithonian"
    ]

    if any(x in s for x in triassic):
        return "Triassic"
    if any(x in s for x in jurassic):
        return "Jurassic"

    return "Unresolved"

def get_original_value(sample_id, column):
    if pd.isna(sample_id) or sample_id not in original.index:
        return np.nan

    value = original.at[sample_id, column]

    if isinstance(value, pd.Series):
        value = value.iloc[0]

    return value

records = []

for _, row in crosswalk.iterrows():
    name = str(row["SampleName"])

    evidence = bed_evidence.get(name)

    if evidence:
        bed_2015, period_2015 = evidence
        match_status = "matched_to_2015_table_3"
    else:
        bed_2015, period_2015 = np.nan, "Not documented"
        match_status = "no_2015_table_3_match"

    sample16 = row["SampleID_2016"]
    sample20 = row["SampleID_2020"]

    target16 = get_original_value(sample16, target_col)
    target20 = get_original_value(sample20, target_col)

    technique16 = get_original_value(sample16, tech_col)
    technique20 = get_original_value(sample20, tech_col)

    period16 = period_from_stage(row["Stage_2016"])
    period20 = period_from_stage(row["Stage_2020"])

    def compare_period(reference, derived):
        if reference == "Not documented":
            return "not independently assessed"
        if reference == "Boundary":
            if derived in ["Triassic", "Jurassic"]:
                return "boundary-compatible; not validated"
            return "unresolved"
        if reference == derived:
            return "agreement"
        if derived == "Unresolved":
            return "unresolved"
        return "disagreement"

    comparison16 = compare_period(period_2015, period16)
    comparison20 = compare_period(period_2015, period20)

    if name == "IYF53D-0":
        external_evidence = (
            "2023 Ikeda et al. identifies this sample as a purple "
            "shale at the base of the post-end-Triassic-extinction "
            "interval using radiolarian biostratigraphy."
        )
        chronology_flag = (
            "independently_supported_boundary_context; "
            "2016 absolute-age/stage assignment requires review"
        )
    else:
        external_evidence = (
            "2015 Table 3 sample-bed crosswalk where available; "
            "no independently verified exact sample age"
        )
        chronology_flag = "absolute chronology unresolved"

    record = row.to_dict()

    record.update({
        "DSMSNI_δ15Nbulk_2016": target16,
        "DSMSNI_δ15Nbulk_2020": target20,
        "DSMSNI_Technique_2016": technique16,
        "DSMSNI_Technique_2020": technique20,
        "2015_source_bed_independent": bed_2015,
        "2015_source_period_independent": period_2015,
        "2015_table3_match_status": match_status,
        "Period_2016_derived": period16,
        "Period_2020_derived": period20,
        "2015_vs_2016_period_review": comparison16,
        "2015_vs_2020_period_review": comparison20,
        "2023_independent_evidence": external_evidence,
        "chronology_review_flag": chronology_flag,
        "sample_identity_caution": (
            "Matching sample identifiers do not independently prove "
            "the same physical aliquot or exact specimen."
        ),
        "age_correction_applied": False,
        "record_deleted": False,
    })

    records.append(record)

register = pd.DataFrame(records)

if len(register) != 26:
    raise ValueError(f"Expected 26 sample pairs; found {len(register)}")

numeric16 = pd.to_numeric(
    register["DSMSNI_δ15Nbulk_2016"], errors="coerce"
)
numeric20 = pd.to_numeric(
    register["DSMSNI_δ15Nbulk_2020"], errors="coerce"
)

matched_2015 = register[
    register["2015_table3_match_status"] == "matched_to_2015_table_3"
]

summary = [
    "EXPERIMENT 56 — INDEPENDENT CHRONOLOGY EVIDENCE AUDIT",
    "=" * 65,
    "",
    f"Total matched sample pairs: {len(register)}",
    f"2015 Table 3 crosswalk matches: {len(matched_2015)}",
    f"Pairs without 2015 Table 3 match: {len(register) - len(matched_2015)}",
    "",
    "DSMS-NI DATABASE ISOTOPE FIELD COVERAGE",
    "-" * 40,
    f"2016 DSMS-NI numeric δ15Nbulk values: {numeric16.notna().sum()}/26",
    f"2020 DSMS-NI numeric δ15Nbulk values: {numeric20.notna().sum()}/26",
    "",
    "CHRONOLOGY",
    "-" * 40,
    f"Median absolute 2016–2020 age difference: "
    f"{pd.to_numeric(register['Age_difference_Ma'], errors='coerce').abs().median():.6f} Ma",
    f"2016 distinct assigned ages: {register['Age_2016'].nunique()}",
    f"2020 distinct assigned ages: {register['Age_2020'].nunique()}",
    "",
    "2015 INDEPENDENT SAMPLE-BED EVIDENCE",
    "-" * 40,
]

for _, r in matched_2015.iterrows():
    summary.append(
        f"{r['SampleName']}: 2015 Bed {int(r['2015_source_bed_independent'])}; "
        f"source period={r['2015_source_period_independent']}; "
        f"2016={r['Period_2016_derived']}; "
        f"2020={r['Period_2020_derived']}"
    )

summary.extend([
    "",
    "SPECIAL CASE: IYF53D-0",
    "-" * 40,
    "This sample is independently identified in the 2023 Ikeda et al. "
    "paper as a purple shale at the base of the post-end-Triassic "
    "extinction interval.",
    "Its 2016 assigned age is 188.05 Ma, while its 2020 assigned age "
    "is 201.40342 Ma.",
    "This is a significant chronology-review candidate, not an "
    "automatically corrected age.",
    "",
    "SCIENTIFIC LIMITATIONS",
    "-" * 40,
    "1. A sample-name match does not prove exact physical aliquot identity.",
    "2. The 2015 bed numbering is a separate numbering system from 2020.",
    "3. Broad period agreement does not validate absolute ages.",
    "4. Boundary compatibility is not proof of correct chronology.",
    "5. No original ages were changed and no observations were deleted.",
    "",
    "OUTPUT",
    "-" * 40,
    str(output_file),
])

register.to_csv(output_file, index=False, encoding="utf-8-sig")
summary_file.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nRegister saved:", output_file)
print("Summary saved:", summary_file)
