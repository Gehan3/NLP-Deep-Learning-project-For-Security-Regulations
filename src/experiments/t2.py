from pathlib import Path
import re

from parser.base_parser import BaseParser

ISO_REGEX = re.compile(
    r"^([5-8]\.\d+)\s+([A-Za-z][A-Za-z0-9\s\-&,().'/]+)$"
)
parser = BaseParser(
    pdf_path=Path("data/raw/ISO 27002.pdf"),
    standard="ISO27002",
    control_regex=ISO_REGEX
)

controls = parser.parse()
print(len(controls))

for c in controls:
    print(c["control_id"], "-", c["title"])

print(f"\nParsed {len(controls)} controls\n")

first = controls[0]

print("=" * 80)
print(first["control_id"])
print(first["title"])
print(f"Page: {first['page']}")
print("=" * 80)

print(first["raw_text"][-1200:])