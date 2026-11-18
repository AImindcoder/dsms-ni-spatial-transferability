from pathlib import Path

path = Path("01_Data/external/83_dotternhausen_original_table2_extracted.txt")
lines = path.read_text(encoding="utf-8").splitlines()

header = next(
    i for i, line in enumerate(lines)
    if "Depth [cm]" in line
)

print("=" * 76)
print("EXPERIMENT 85: TABLE 2 READING-ORDER INSPECTION")
print("=" * 76)

print("\nFIRST TABLE HEADER AND FOLLOWING 115 LINES:\n")

for i in range(header, min(header + 116, len(lines))):
    value = lines[i].strip()
    print(f"{i + 1:04d} | {value!r}")

print("\nOTHER TABLE HEADER POSITIONS:")
for i, line in enumerate(lines):
    if "Depth [cm]" in line:
        print(f"Line {i + 1}: {line.strip()}")

print("\nEXPERIMENT 85 COMPLETED")
