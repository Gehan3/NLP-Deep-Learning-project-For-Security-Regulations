import re
from pathlib import Path
import fitz

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FOLDER = BASE_DIR / "data" / "raw"

# ==========================================================
# Inspector
# ==========================================================

def inspect_pdf(pdf_path: Path):

    print("\n" + "=" * 100)
    print(f"Inspecting: {pdf_path.name}")
    print("=" * 100)

    doc = fitz.open(pdf_path)

    total_pages = len(doc)

    print(f"Pages: {total_pages}")

    heading_count = 0

    for page_num, page in enumerate(doc):

        text = page.get_text()

        if not text.strip():
            continue

        lines = text.splitlines()

        interesting_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------------
            # Print anything that LOOKS like a heading
            # ------------------------------------------------------

            if re.match(r"^\d", line):
                interesting_lines.append(line)

            elif line.lower().startswith("purpose"):
                interesting_lines.append(line)

            elif line.lower().startswith("implementation"):
                interesting_lines.append(line)

            elif line.lower().startswith("additional"):
                interesting_lines.append(line)

            elif line.lower().startswith("attributes"):
                interesting_lines.append(line)

        if interesting_lines:

            print(f"\n{'='*40}")
            print(f"PAGE {page_num + 1}")
            print(f"{'='*40}")

            for line in interesting_lines:

                heading_count += 1
                print(line)

    print("\n" + "=" * 100)
    print(f"Interesting lines found: {heading_count}")
    print("=" * 100)

    doc.close()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    pdfs = list(DATA_FOLDER.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found.")
        exit()

    for pdf in pdfs:
        inspect_pdf(pdf)