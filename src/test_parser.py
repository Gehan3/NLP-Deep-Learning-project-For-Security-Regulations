from pathlib import Path
import re

from parser.base_parser import BaseParser, SECTION_MAP

CONTROL_REGEX = re.compile(
    r"^([5-8]\.\d+)\s+([A-Z][A-Za-z0-9 ,()'\/\-:&]+)$"
)

parser = BaseParser(
    pdf_path=Path("data/raw/ISO 27002.pdf"),
    standard="ISO27002",
    control_regex=CONTROL_REGEX,
    section_map=SECTION_MAP
)

controls = parser.parse()

print(f"\nFound {len(controls)} controls\n")

print("=" * 80)

for control in controls:

    print(control["control_id"], "-", control["title"])