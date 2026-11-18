from pathlib import Path
from urllib.request import Request, urlopen
import json

record_id = "4778565"
url = f"https://zenodo.org/api/records/{record_id}"

request = Request(url, headers={"User-Agent": "DSMS-NI-research-audit/1.0"})

print("=" * 74)
print("EXPERIMENT 75: DOTTERNHAUSEN ORIGINAL DATA REPOSITORY INSPECTION")
print("=" * 74)

with urlopen(request, timeout=60) as response:
    record = json.loads(response.read().decode("utf-8"))

metadata = record.get("metadata", {})

print("\nREPOSITORY")
print("Title:", metadata.get("title"))
print("DOI:", metadata.get("doi"))
print("Publication date:", metadata.get("publication_date"))
print("Description:", str(metadata.get("description", ""))[:1200])

files_data = record.get("files", [])

if isinstance(files_data, dict):
    files = files_data.get("entries", [])
elif isinstance(files_data, list):
    files = files_data
else:
    raise TypeError(f"Unexpected files structure: {type(files_data)}")

if not files:
    raise RuntimeError("No files found in Zenodo record.")

rows = []

for item in files:
    links = item.get("links", {})
    row = {
        "filename": item.get("key") or item.get("filename"),
        "size_bytes": item.get("size"),
        "checksum": item.get("checksum"),
        "download_url": links.get("self") or links.get("download"),
    }
    rows.append(row)

    print("\nFILE:", row["filename"])
    print("Size:", row["size_bytes"])
    print("Checksum:", row["checksum"])
    print("Download:", row["download_url"])

output = Path("01_Data/external/Dotternhausen_Zenodo_file_inventory.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(rows, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nINVENTORY:", output)
print("FILES FOUND:", len(rows))
print("\nNo files downloaded and original dataset modified.")
print("EXPERIMENT 75 COMPLETED")
