from pathlib import Path
import pandas as pd

root = Path("01_Data")

print("=" * 76)
print("EXPERIMENT 93: SOURCE FILE DISCOVERY")
print("=" * 76)

for folder in ["external", "interim", "processed", "raw"]:
    directory = root / folder
    print(f"\n--- {folder.upper()} ---")

    if not directory.exists():
        print("Directory not found")
        continue

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file()
        and any(term in p.name.lower()
                for term in ["dotternhausen", "wang", "table1", "table2", "dsms"])
    )

    for p in files:
        print(f"{p} | {p.stat().st_size:,} bytes")

csv_path = root / "interim" / "dotternhausen_table2_coordinate_reconstruction.csv"

print("\n--- RECONSTRUCTED TABLE 2 ---")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst five records:")
    print(df.head().to_string(index=False))
else:
    print("Reconstruction CSV not found")

print("\nEXPERIMENT 93 COMPLETED")
