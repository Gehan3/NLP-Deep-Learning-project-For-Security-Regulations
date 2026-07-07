from enum import Enum
from pathlib import Path
import fitz
import re
import json


# ==========================================================
# Parser States
# ==========================================================

class ParserState(Enum):
    NONE = 0
    GENERAL = 1
    PURPOSE = 2
    IMPLEMENTATION = 3
    ADDITIONAL = 4
    ATTRIBUTES = 5
    TESTING = 6


# ==========================================================
# Generic Section Mapping
# ==========================================================

SECTION_MAP = {
    "Purpose": ParserState.PURPOSE,
    "Implementation guidance": ParserState.IMPLEMENTATION,
    "Additional information": ParserState.ADDITIONAL,
    "Attributes": ParserState.ATTRIBUTES,
}


# ==========================================================
# State -> JSON Field
# ==========================================================

STATE_TO_FIELD = {
    ParserState.GENERAL: "general_text",
    ParserState.PURPOSE: "purpose",
    ParserState.IMPLEMENTATION: "implementation_guidance",
    ParserState.ADDITIONAL: "additional_information",
    ParserState.ATTRIBUTES: "attributes",
    ParserState.TESTING: "testing_procedure",
}


# ==========================================================
# Empty Control Factory
# ==========================================================

def new_control():

    return {
        "standard": "",
        "control_id": "",
        "parent_section": "",
        "title": "",
        "page": 0,

        "general_text": [],
        "purpose": [],
        "implementation_guidance": [],
        "additional_information": [],
        "attributes": [],
        "testing_procedure": []
    }


# ==========================================================
# Base Parser
# ==========================================================

class BaseParser:

    def __init__(
        self,
        pdf_path: Path,
        standard: str,
        control_regex,
        section_map: dict
    ):

        self.pdf_path = pdf_path
        self.standard = standard
        self.control_regex = control_regex
        self.section_map = section_map

        self.state = ParserState.NONE
        self.current = None
        self.controls = []

    # ======================================================

    def extract_lines(self):

     doc = fitz.open(self.pdf_path)

     lines = []

     for page_num, page in enumerate(doc):

        text = page.get_text()

        for line in text.splitlines():

            # Normalize whitespace (tabs, multiple spaces, etc.)
            line = " ".join(line.strip().split())

            if line:

                lines.append({
                    "page": page_num + 1,
                    "text": line
                })

     doc.close()

     return lines  

    # ======================================================

    def start_new_control(
        self,
        control_id,
        title,
        page
    ):

        self.current = new_control()

        self.current["standard"] = self.standard
        self.current["control_id"] = control_id
        self.current["parent_section"] = control_id.split(".")[0]
        self.current["title"] = title
        self.current["page"] = page

        self.state = ParserState.GENERAL

    # ======================================================

    def save_current_control(self):

        if self.current is None:
            return

        for key, value in self.current.items():

            if isinstance(value, list):

                self.current[key] = "\n".join(value).strip()

        self.controls.append(self.current)

        self.current = None

    # ======================================================

    def parse(self):

        lines = self.extract_lines()

        parsing_started = False

        for line in lines:

            text = line["text"]
            page = line["page"]

            # --------------------------------------------------
            # Wait until controls begin
            # --------------------------------------------------

            if not parsing_started:

                if text.startswith("5 Organizational controls"):

                    parsing_started = True

                    print(f"Started parsing on page {page}")

                continue

            # --------------------------------------------------
            # Detect a new control
            # --------------------------------------------------

            match = self.control_regex.match(text)

            if match:

                control_id = match.group(1)
                title = match.group(2)

                self.save_current_control()

                self.start_new_control(
                    control_id,
                    title,
                    page
                )

                continue

            # --------------------------------------------------
            # Ignore everything until first control
            # --------------------------------------------------

            if self.current is None:
                continue

            # --------------------------------------------------
            # Detect section headers
            # --------------------------------------------------

            if text in self.section_map:

                self.state = self.section_map[text]

                continue

            # --------------------------------------------------
            # Store text
            # --------------------------------------------------

            field = STATE_TO_FIELD.get(self.state)

            if field:

                self.current[field].append(text)

        # Save final control

        self.save_current_control()

        return self.controls

    # ======================================================

    def save_json(self, output_path: Path):

        with open(output_path, "w", encoding="utf-8") as f:

            json.dump(
                self.controls,
                f,
                indent=4,
                ensure_ascii=False
            )