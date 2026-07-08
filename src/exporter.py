import json
from pathlib import Path

from parser.base_parser import BaseParser
import re


ISO_REGEX = re.compile(
    r"^([5-8]\.\d+)\s+([A-Za-z][A-Za-z0-9\s\-&,().'/]+)$"
)


parser = BaseParser(
    pdf_path=Path("data/raw/ISO 27002.pdf"),
    standard="ISO27002",
    control_regex=ISO_REGEX
)


controls = parser.parse()


output_path = Path(
    "data/processed/iso27002_controls2.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        controls,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    f"Saved {len(controls)} controls to {output_path}"
)