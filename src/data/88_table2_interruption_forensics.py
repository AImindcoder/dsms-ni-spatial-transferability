from pathlib import Path

path = Path("01_Data/external/83_dotternhausen_original_table2_extracted.txt")
lines = path.read_text(encoding="utf-8").splitlines()

print("=" * 76)
print("EXPERIMENT 88: TABLE INTERRUPTION FORENSICS")
print("=" * 76)

for start, end in [(500, 585), (580, 640), (870, 915)]:
    print(f"\n--- LINES {start + 1} TO {end} ---")
    for i in range(start, min(end, len(lines))):
        print(f"{i + 1:04d} | {lines[i].strip()!r}")

print("\nEXPERIMENT 88 COMPLETED")
