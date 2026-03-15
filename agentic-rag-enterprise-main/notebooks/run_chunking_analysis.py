"""
Standalone runner for chunking_analysis.ipynb — use this to verify the notebook
logic without needing nbconvert. Run: python notebooks/run_chunking_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import faiss
import matplotlib
matplotlib.use('Agg')  # headless — saves PNG without needing a display
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import SectionAwareChunker, Chunk

PDF_DIR   = 'data/sample_pdfs'
TEST_PATH = 'src/evaluation/test_set.json'
TOP_K     = 5

# ── Parse PDFs ──────────────────────────────────────────────────────────────
print('Parsing PDFs...')
parser    = PDFParser(remove_headers_footers=True)
documents = parser.parse_directory(PDF_DIR)
print(f'  {len(documents)} docs, {sum(d.total_pages for d in documents)} pages\n')

with open(TEST_PATH) as f:
    test_set = json.load(f)

# ── Embed model (shared across strategies) ───────────────────────────────────
print('Loading embedding model...')
model = SentenceTransformer('all-MiniLM-L6-v2')

# ── Fixed-size chunker ───────────────────────────────────────────────────────
def fixed_size_chunks(documents, chunk_words: int, overlap_words: int = 32):
    chunks = []
    for doc in documents:
        for page in doc.pages:
            words = page.text.split()
            start, idx = 0, 0
            while start < len(words):
                end  = min(start + chunk_words, len(words))
                text = ' '.join(words[start:end])
                if len(text) >= 80:
                    chunks.append(Chunk(
                        text=text,
                        metadata={'page_number': page.page_number, 'filename': doc.filename},
                        chunk_id=f'{doc.filename}__f{chunk_words}_{page.page_number}_{idx:03d}',
                    ))
                    idx += 1
                start += chunk_words - overlap_words
    return chunks

# ── FAISS helpers ────────────────────────────────────────────────────────────
def build_index(chunks):
    texts = [c.text for c in chunks]
    embs  = model.encode(texts, batch_size=64, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def hit_rate(index, chunks, test_set, top_k=TOP_K):
    hits = []
    for t in test_set:
        q_emb = model.encode([t['question']], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q_emb)
        _, idxs = index.search(q_emb, min(top_k, len(chunks)))
        pages   = [chunks[i].metadata['page_number'] for i in idxs[0] if i != -1]
        hits.append(bool(set(pages) & set(t.get('expected_pages', []))))
    return sum(hits) / len(hits), hits

# ── Run all three strategies ─────────────────────────────────────────────────
results = {}

for label, chunks in [
    ('Fixed-256',     fixed_size_chunks(documents, chunk_words=256, overlap_words=32)),
    ('Fixed-512',     fixed_size_chunks(documents, chunk_words=512, overlap_words=64)),
    ('Section-Aware', [c for doc in documents
                       for c in SectionAwareChunker(chunk_size=512, chunk_overlap=64, min_chunk_size=100)
                                   .chunk_document(doc.pages, doc.metadata)]),
]:
    print(f'[{label}] {len(chunks)} chunks — embedding...', end=' ', flush=True)
    index = build_index(chunks)
    rate, hits = hit_rate(index, chunks, test_set)
    results[label] = {'rate': rate, 'hits': hits, 'n_chunks': len(chunks)}
    print(f'hit rate = {rate:.1%}')

# ── Summary ──────────────────────────────────────────────────────────────────
print('\nSummary:')
for name, r in results.items():
    print(f'  {name:20s}: {r["rate"]:.1%}  ({r["n_chunks"]} chunks)')

# ── Bar chart ────────────────────────────────────────────────────────────────
strategies = list(results.keys())
rates      = [results[s]['rate'] * 100 for s in strategies]
n_chunks   = [results[s]['n_chunks'] for s in strategies]
colors     = ['#6baed6', '#3182bd', '#08519c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(strategies, rates, color=colors, width=0.5, zorder=3)
ax.set_ylabel('Retrieval Hit Rate (%)', fontsize=12)
ax.set_ylim(0, 115)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

for bar, rate, n in zip(bars, rates, n_chunks):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f'{rate:.1f}%\n({n} chunks)',
        ha='center', va='bottom', fontsize=10, fontweight='bold',
    )

ax.set_title('Chunking Strategy vs Retrieval Hit Rate\n(top-5 retrieval, 15 test queries)', fontsize=13)
ax.set_xlabel('Chunking Strategy', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = 'data/chunking_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nChart saved to {out}')

# ── Per-query breakdown ──────────────────────────────────────────────────────
print(f'\n{"Question":<55} {"F-256":>6} {"F-512":>6} {"Sect":>6}')
print('-' * 76)
h256 = results['Fixed-256']['hits']
h512 = results['Fixed-512']['hits']
hsa  = results['Section-Aware']['hits']
for i, t in enumerate(test_set):
    q = t['question'][:53]
    print(f'{q:<55} {"HIT" if h256[i] else "miss":>6} {"HIT" if h512[i] else "miss":>6} {"HIT" if hsa[i] else "miss":>6}')
