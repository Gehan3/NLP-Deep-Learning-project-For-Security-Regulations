from enum import Enum
from pathlib import Path
import fitz


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
# Section Map for ISO 27001
# ==========================================================
SECTION_MAP = {
    "5": "Leadership",
    "6": "Planning",
    "7": "Support",
    "8": "Operation",
    "9": "Performance evaluation",
    "10": "Improvement"
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

        "raw_text": []
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
        section_map=None
    ):

        self.pdf_path = pdf_path
        self.standard = standard
        self.control_regex = control_regex
        self.section_map = section_map
        self.current = None

        self.controls = []

    # ------------------------------------------------------

    def extract_lines(self):

        doc = fitz.open(self.pdf_path)

        lines = []

        for page_num, page in enumerate(doc):

            text = page.get_text()

            for line in text.splitlines():

                line = line.strip()

                if line:

                    lines.append(
                        {
                            "page": page_num + 1,
                            "text": line
                        }
                    )

        doc.close()

        return lines

    # ------------------------------------------------------

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

    # ------------------------------------------------------

    def save_current_control(self):

        if self.current is None:
            return

        self.current["raw_text"] = "\n".join(
            self.current["raw_text"]
        ).strip()

        self.controls.append(self.current)

        self.current = None

    # ------------------------------------------------------

    def parse(self):

        lines = self.extract_lines()

        started = False

        for line in lines:

            text = line["text"]

            match = self.control_regex.match(text)

            if match:

                control_id = match.group(1)

                title = match.group(2)

                # Ignore introduction, TOC, annexes, etc.
                try:
                    major = int(control_id.split(".")[0])

                    if major < 5:
                        continue

                except:
                    continue

                started = True

                self.save_current_control()

                self.start_new_control(
                    control_id,
                    title,
                    line["page"]
                )

                continue

            if not started:
                continue

            if self.current is not None:

                self.current["raw_text"].append(text)

        self.save_current_control()

        return self.controls