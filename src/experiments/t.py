from pathlib import Path
import re

from parser.base_parser import BaseParser


CONTROL_REGEX = re.compile(
    r"^(\d+\.\d+)\s+(.+)$"
)

parser = BaseParser(
    pdf_path=Path("data/raw/ISO 27002.pdf"),
    standard="ISO27002",
    control_regex=CONTROL_REGEX,
    section_map={}
)

lines = parser.extract_lines()

controls = []

for line in lines:

    match = CONTROL_REGEX.match(line["text"])

    if match:

        control_id = match.group(1)
        title = match.group(2)

        controls.append(
            (control_id, title, line["page"])
        )


print(f"\nFound {len(controls)} controls\n")

print("="*80)

for control in controls[:40]:

    print(control)