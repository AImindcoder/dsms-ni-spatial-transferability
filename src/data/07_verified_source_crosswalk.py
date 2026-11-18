import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

target_dois = [
    "10.1038/377514a0",
    "10.1038/378216a0",
    "10.1029/2006gl025864",
    "10.1016/j.quascirev.2006.04.009"
]

for target in target_dois:
    subset = df[
        df["DOI"].astype("string").str.strip().str.lower() == target
    ]

    print("\n" + "=" * 75)
    print("DOI:", target)
    print("Record count:", len(subset))
    print("Titles:")
    print(subset["Title"].value_counts().to_string())
    print("\nReferences:")
    print(subset["Reference"].value_counts().head(5).to_string())
    print("\nAuthors:", subset["FirstAuthor"].value_counts().to_dict())
    print("Years:", subset["Year"].value_counts().to_dict())
