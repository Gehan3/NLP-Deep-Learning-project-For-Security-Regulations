import json
import re
from pathlib import Path

INPUT_FILE = Path("data/processed/iso27002_controls2.json")
OUTPUT_FILE = Path("data/cleaned/iso27002_cleaned.json")


def clean_text(text):

    if not text:
        return ""

    text = str(text)

    # PDF artifacts
    text = text.replace("\ufeff", "")
    text = text.replace("\x0c", "")
    text = text.replace("\x08", "")
    text = text.replace("\u00a0", " ")

    # Remove ISO footer
    text = re.sub(
        r"© ISO/IEC.*?All rights reserved",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove page markers
    text = re.sub(
        r"\n\s*\d+\s*\nISO/IEC 27002:2022\(E\)",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_sections(raw):

    raw = clean_text(raw)

    # Remove everything before the actual Control section
    match = re.search(
        r"\nControl\n",
        raw
    )

    if match:
        raw = raw[match.start() + 1:]

    sections = {
        "control": "",
        "purpose": "",
        "guidance": "",
        "other_information": ""
    }

    control_match = re.search(r"\bControl\b", raw)
    purpose_match = re.search(r"\bPurpose\b", raw)
    guidance_match = re.search(r"\bGuidance\b", raw)
    other_match = re.search(r"\bOther information\b", raw)

    if not control_match:
        return sections

    control_start = control_match.end()

    if purpose_match:
        sections["control"] = raw[
            control_start:purpose_match.start()
        ].strip()

    if purpose_match and guidance_match:
        sections["purpose"] = raw[
            purpose_match.end():guidance_match.start()
        ].strip()

    if guidance_match and other_match:
        sections["guidance"] = raw[
            guidance_match.end():other_match.start()
        ].strip()

    elif guidance_match:
        sections["guidance"] = raw[
            guidance_match.end():
        ].strip()

    if other_match:
        sections["other_information"] = raw[
            other_match.end():
        ].strip()

    for key in sections:
        sections[key] = clean_text(sections[key])

    return sections


def clean_controls():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        controls = json.load(f)

    cleaned = []

    for c in controls:

        sections = split_sections(c["raw_text"])

        cleaned.append({

            "standard": c["standard"],

           "control_id": str(c["control_id"]).strip(),#عupdate to remove zero from retrieval

            "title": c["title"],

            "page": c["page"],

            "control": sections["control"],

            "purpose": sections["purpose"],

            "guidance": sections["guidance"],

            "other_information": sections["other_information"]

        })

    return cleaned


if __name__ == "__main__":

    cleaned = clean_controls()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            cleaned,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Cleaned {len(cleaned)} controls")

    print("\nExample:\n")
    print(
        json.dumps(
            cleaned[0],
            indent=4,
            ensure_ascii=False
        )
    )