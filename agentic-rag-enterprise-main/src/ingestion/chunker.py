"""
Section-Aware Chunker — splits documents respecting structure.
Unlike fixed-size chunking, this keeps tables intact and
splits on section headers and paragraph boundaries.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 0)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")


class SectionAwareChunker:
    """Chunk documents by section structure, not fixed token count."""

    # Common section header patterns
    HEADER_PATTERNS = [
        r"^#{1,6}\s+.+",                           # Markdown headers
        r"^[A-Z][A-Za-z\s]{2,50}$",               # Title Case lines (short)
        r"^\d+\.\s+[A-Z].+",                       # Numbered sections: 1. Title
        r"^\d+\.\d+\s+[A-Z].+",                   # Sub-sections: 1.1 Title
        r"^(ARTICLE|SECTION|CHAPTER)\s+\d+",       # Legal document sections
        r"^(Introduction|Conclusion|Summary|Abstract|Overview|Background)",
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
        respect_sections: bool = True,
        keep_tables_intact: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sections = respect_sections
        self.keep_tables_intact = keep_tables_intact

    def chunk_document(self, pages: list, doc_metadata: dict = None) -> list[Chunk]:
        """Chunk a parsed document into retrieval-ready pieces."""
        doc_metadata = doc_metadata or {}
        all_chunks = []
        current_section = "Document Start"

        for page in pages:
            page_chunks = self._chunk_page(
                text=page.text,
                page_number=page.page_number,
                tables=page.tables,
                current_section=current_section,
                doc_metadata=doc_metadata,
            )
            all_chunks.extend(page_chunks)

            # Track current section across pages
            for chunk in page_chunks:
                if chunk.metadata.get("section"):
                    current_section = chunk.metadata["section"]

        # Assign chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{doc_metadata.get('filename', 'doc')}__chunk_{i:04d}"

        return all_chunks

    def _chunk_page(
        self,
        text: str,
        page_number: int,
        tables: list,
        current_section: str,
        doc_metadata: dict,
    ) -> list[Chunk]:
        """Chunk a single page."""
        chunks = []

        # First, handle tables as standalone chunks
        if self.keep_tables_intact and tables:
            for table in tables:
                table_text = self._table_to_text(table)
                if len(table_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=table_text,
                        metadata={
                            "page_number": page_number,
                            "section": current_section,
                            "content_type": "table",
                            **doc_metadata,
                        },
                    ))

        # Split remaining text by sections
        if self.respect_sections:
            sections = self._split_by_sections(text)
        else:
            sections = [("", text)]

        for section_title, section_text in sections:
            if section_title:
                current_section = section_title

            # Split large sections into paragraph-based chunks
            paragraphs = self._split_paragraphs(section_text)
            current_chunk_parts = []
            current_length = 0

            for para in paragraphs:
                para_len = len(para.split())  # rough word count

                if current_length + para_len > self.chunk_size and current_chunk_parts:
                    # Emit current chunk
                    chunk_text = "\n\n".join(current_chunk_parts)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text,
                            metadata={
                                "page_number": page_number,
                                "section": current_section,
                                "content_type": "text",
                                **doc_metadata,
                            },
                        ))

                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_chunk_parts:
                        overlap_text = current_chunk_parts[-1]
                        current_chunk_parts = [overlap_text, para]
                        current_length = len(overlap_text.split()) + para_len
                    else:
                        current_chunk_parts = [para]
                        current_length = para_len
                else:
                    current_chunk_parts.append(para)
                    current_length += para_len

            # Don't forget the last chunk
            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            "page_number": page_number,
                            "section": current_section,
                            "content_type": "text",
                            **doc_metadata,
                        },
                    ))

        return chunks

    def _split_by_sections(self, text: str) -> list[tuple[str, str]]:
        """Split text into (section_title, section_body) pairs."""
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_body = []

        for line in lines:
            is_header = any(
                re.match(pattern, line.strip())
                for pattern in self.HEADER_PATTERNS
            )

            if is_header and line.strip():
                # Save previous section
                if current_body:
                    sections.append((current_title, "\n".join(current_body).strip()))
                current_title = line.strip()
                current_body = []
            else:
                current_body.append(line)

        # Last section
        if current_body:
            sections.append((current_title, "\n".join(current_body).strip()))

        return sections if sections else [("", text)]

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving meaningful blocks."""
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _table_to_text(self, table: dict) -> str:
        """Convert table dict to readable text format."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        lines = []
        if headers:
            lines.append("| " + " | ".join(str(h or "") for h in headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")

        for row in rows:
            lines.append("| " + " | ".join(str(c or "") for c in row) + " |")

        return "\n".join(lines)
