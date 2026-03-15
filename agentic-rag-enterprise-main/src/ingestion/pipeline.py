"""
Ingestion Pipeline — end-to-end: PDFs → parsed → chunked → embedded → stored.
Run: python -m src.ingestion.pipeline --pdf-dir data/sample_pdfs/
"""
import argparse
from pathlib import Path

from .pdf_parser import PDFParser
from .chunker import SectionAwareChunker
from ..retrieval.vector_store import VectorStore


def run_ingestion(pdf_dir: str, chroma_dir: str = "./data/chroma_db"):
    """Ingest all PDFs from a directory into the vector store."""
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    # Step 1: Parse PDFs
    print("\n[1/3] Parsing PDFs...")
    parser = PDFParser(remove_headers_footers=True)
    documents = parser.parse_directory(pdf_dir)
    print(f"  Parsed {len(documents)} documents")

    # Step 2: Chunk documents
    print("\n[2/3] Chunking documents...")
    chunker = SectionAwareChunker(
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=100,
        respect_sections=True,
        keep_tables_intact=True,
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc.pages, doc.metadata)
        all_chunks.extend(chunks)
        print(f"  {doc.filename}: {len(chunks)} chunks")

    print(f"  Total chunks: {len(all_chunks)}")

    # Step 3: Embed and store
    print("\n[3/3] Embedding and storing...")
    store = VectorStore(persist_dir=chroma_dir)

    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    ids = [c.chunk_id for c in all_chunks]

    store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
    print(f"  Stored {len(all_chunks)} chunks in ChromaDB")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    return all_chunks


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Ingest PDFs into vector store")
    arg_parser.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    arg_parser.add_argument("--chroma-dir", default="./data/chroma_db", help="ChromaDB persist directory")
    args = arg_parser.parse_args()

    run_ingestion(args.pdf_dir, args.chroma_dir)
