import pandas as pd
from pathlib import Path

source = Path("02_Data_Audit/provenance_analysis/doi_provenance_summary.csv")
output = Path("02_Data_Audit/provenance_analysis/metadata_review_decision_register.csv")

df = pd.read_csv(source)

decisions = {
    "10.1016/j.chemgeo.2010.09.001": (
        "formatting_variation",
        "Same author and year; title differs by hyphen/en-dash formatting.",
        "Normalize for comparison; retain original metadata."
    ),
    "10.1016/j.orggeochem.2022.104451": (
        "formatting_variation",
        "Same author and year; title differs by whitespace/newline.",
        "Normalize for comparison; retain original metadata."
    ),
    "10.1029/1999pa000422": (
        "formatting_variation",
        "Title differs by punctuation, capitalization and hyphen formatting.",
        "Normalize for comparison; retain original metadata."
    ),
    "10.1029/2004gl020511": (
        "probable_year_error",
        "Three records show Year 2013 while DOI, title, author and reference identify Higginson 2004.",
        "Flag three records for source verification; do not overwrite raw year."
    ),
    "10.1029/2004pa001114": (
        "possible_source_attribution_error",
        "One Schlung 2013 record appears in a DOI group otherwise identified as Robinson 2005.",
        "Investigate the individual record and original source before assigning its study group."
    ),
    "10.1029/2006pa001308": (
        "author_name_formatting",
        "Author variants differ by Unicode hyphen formatting.",
        "Normalize author name for comparison; retain original metadata."
    ),
    "10.1029/2008pa001702": (
        "probable_title_contamination",
        "One record contains an unrelated title appended to the main Robinson 2009 title.",
        "Flag the individual record for source verification; preserve original title."
    ),
    "10.1029/93pa00784": (
        "formatting_variation",
        "Title differences appear to involve capitalization, punctuation and superscript formatting.",
        "Normalize for comparison; retain original metadata."
    ),
    "10.1111/j.1365-2117.2009.00427.x": (
        "possible_title_variant",
        "Some records contain an additional descriptive title suffix.",
        "Check the publication title; provisionally retain one DOI group."
    ),
    "10.19658/j.issn.1007-2802.2023.42.084": (
        "language_title_variant",
        "English and Chinese title variants appear under the same DOI and author/year.",
        "Treat as a likely bilingual metadata variation; retain original metadata."
    )
}

records = []

for doi, (category, reason, action) in decisions.items():
    match = df[df["_doi_normalized"].astype(str) == doi]

    if match.empty:
        print("WARNING: DOI not found:", doi)
        continue

    row = match.iloc[0]

    records.append({
        "doi": doi,
        "record_count": int(row["record_count"]),
        "unique_titles": int(row["unique_titles"]),
        "unique_authors": int(row["unique_authors"]),
        "unique_years": int(row["unique_years"]),
        "review_classification": category,
        "evidence": reason,
        "recommended_action": action,
        "decision_status": "provisional_requires_review",
        "raw_data_modified": False
    })

register = pd.DataFrame(records)
register.to_csv(output, index=False, encoding="utf-8-sig")

print("METADATA REVIEW DECISION REGISTER")
print("=" * 65)
print("Review groups:", len(register))
print("Observations represented:", register["record_count"].sum())
print("\nCLASSIFICATION COUNTS")
print(register["review_classification"].value_counts().to_string())
print("\nREGISTER")
print(register[[
    "doi", "record_count", "review_classification", "decision_status"
]].to_string(index=False))
print("\nSaved:", output)
