import fitz  # PyMuPDF
from pathlib import Path


DATA_FOLDER = Path("data/raw")


def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    print("=" * 80)
    print(f"Reading: {pdf_path.name}")
    print(f"Pages: {len(doc)}")
    print("=" * 80)

    for page_number, page in enumerate(doc):

        text = page.get_text()

        print(f"\nPAGE {page_number + 1}")
        print("-" * 80)

        print(text[:1500])  # only first 1500 chars for now

        if page_number == 5:
            break

    doc.close()


if __name__ == "__main__":

    for pdf in DATA_FOLDER.glob("*.pdf"):
        read_pdf(pdf)