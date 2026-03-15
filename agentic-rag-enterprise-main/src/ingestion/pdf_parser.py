"""
PDF Parser — extracts structured text from PDFs.
Handles tables, multi-column layouts, headers/footers.
"""
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    pages: list[PageContent]
    total_pages: int
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """Extract structured content from PDF files."""

    def __init__(self, remove_headers_footers: bool = True):
        self.remove_headers_footers = remove_headers_footers

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file into structured content."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_content = self._extract_page(page, page_num + 1)
            pages.append(page_content)

        # Extract document-level metadata
        meta = doc.metadata or {}
        doc_metadata = {
            "title": meta.get("title", pdf_path.stem),
            "author": meta.get("author", ""),
            "total_pages": len(doc),
            "filename": pdf_path.name,
        }

        doc.close()

        return ParsedDocument(
            filename=pdf_path.name,
            pages=pages,
            total_pages=len(pages),
            metadata=doc_metadata,
        )

    def _extract_page(self, page: fitz.Page, page_number: int) -> PageContent:
        """Extract text and tables from a single page."""
        # Get page dimensions for header/footer detection
        rect = page.rect
        header_zone = rect.height * 0.08  # top 8%
        footer_zone = rect.height * 0.92  # bottom 8%

        # Extract text blocks with position info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        text_parts = []
        tables = []

        for block in blocks:
            if block["type"] != 0:  # skip image blocks
                continue

            # Filter headers/footers based on position
            if self.remove_headers_footers:
                block_y = block["bbox"][1]
                block_y_bottom = block["bbox"][3]
                if block_y < header_zone or block_y_bottom > footer_zone:
                    continue

            # Extract text from block
            block_text = ""
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span["text"]
                block_text += line_text.strip() + "\n"

            block_text = block_text.strip()
            if block_text:
                text_parts.append(block_text)

        # Try to extract tables using PyMuPDF's table detection
        try:
            page_tables = page.find_tables()
            for table in page_tables:
                table_data = table.extract()
                if table_data:
                    tables.append({
                        "headers": table_data[0] if table_data else [],
                        "rows": table_data[1:] if len(table_data) > 1 else [],
                        "bbox": list(table.bbox),
                    })
        except Exception:
            pass  # Table detection can fail on some PDFs

        # Combine text
        full_text = "\n\n".join(text_parts)

        return PageContent(
            page_number=page_number,
            text=full_text,
            tables=tables,
            metadata={"has_tables": len(tables) > 0},
        )

    def parse_directory(self, dir_path: str | Path) -> list[ParsedDocument]:
        """Parse all PDFs in a directory."""
        dir_path = Path(dir_path)
        documents = []
        for pdf_file in sorted(dir_path.glob("*.pdf")):
            print(f"Parsing: {pdf_file.name}")
            try:
                doc = self.parse(pdf_file)
                documents.append(doc)
                print(f"  -> {doc.total_pages} pages extracted")
            except Exception as e:
                print(f"  -> ERROR: {e}")
        return documents
