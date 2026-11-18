from pathlib import Path

path = Path("01_Data/external/83_dotternhausen_original_table2_extracted.txt")
lines = path.read_text(encoding="utf-8").splitlines()

print("=" * 72)
print("EXPERIMENT 86: TABLE 2 PAGE-BOUNDARY AUDIT")
print("=" * 72)

for start, end in [(435, 480), (895, 945), (1030, 1072)]:
    print(f"\n--- LINES {start + 1} TO {min(end, len(lines))} ---")
    for i in range(start, min(end, len(lines))):
        print(f"{i + 1:04d} | {lines[i].strip()!r}")

print("\nEXPERIMENT 86 COMPLETED")
