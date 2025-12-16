import json
from pathlib import Path

# Get the docs directory (where this script is located)
docs_dir = Path(__file__).parent

# Find all HTML files except index.html
html_files = [
    f.name for f in docs_dir.glob("*.html")
    if f.name != "index.html"
]

# Sort the files naturally (handles numbers in filenames)
html_files.sort(key=lambda x: x.lower())

# Write to pages.json
output_file = docs_dir / "pages.json"
with open(output_file, "w") as f:
    json.dump(html_files, f, indent=2)

print(f"Found {len(html_files)} HTML files")
print(f"Saved to {output_file}")
