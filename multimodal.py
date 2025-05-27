#!/usr/bin/env python3
# coding: utf-8
#
# Multimodal RAG — full pipeline for text, table, and image extraction,
# with table captions, image summaries, and robust vector indexing.
# Date: May 2025

# ─── 0. Dependencies (install if needed) ─────────────────────────────
!pip install --upgrade \
google-generativeai docling transformers pillow \
langchain langchain_community chromadb faiss-cpu tqdm \
sentence-transformers pymupdf pytesseract tiktoken

# ─── 1. Imports & Helpers ───────────────────────────────────────────
import re, difflib, sqlite3, shutil, warnings, getpass
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageChops
import fitz, pytesseract, tiktoken, google.generativeai as genai
from IPython.display import display, Markdown

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=UserWarning)

# Tokenizer + token counter
enc = tiktoken.get_encoding("cl100k_base")
def n_tok(txt: str) -> int:
    return len(enc.encode(txt))

# Roman numerals → int (for label normalization)
ROMAN_MAP = {'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000}
def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        val = ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total

# Normalize "Figure IIa" or "Table 3b" → "fig2a" / "table3b"
LABEL_RE = re.compile(r'^(fig(?:ure)?|table)\s*([IVXLCDM]+|\d+)([a-z])?[\.\s]*[:\-]?', re.I)
def label_key(txt: str) -> str:
    if not txt:
        return None
    s = txt.lower().replace(' ', '')
    m = LABEL_RE.match(s)
    if not m:
        return None
    head, raw, suffix = m.groups()
    num = raw if raw.isdigit() else str(roman_to_int(raw))
    key = ('fig' if head.startswith('fig') else 'table') + num
    if suffix:
        key += suffix
    return key

# Regex to detect existing fig/table labels in exported markdown
FIG_TBL = re.compile(r'(fig(?:ure)?\.?\s*\d+[a-z]?|table\s+[IVXLCDM\d]+[a-z]?)', re.I)
def canon(lbl: str) -> str:
    return FIG_TBL.sub(lambda m: re.sub(r'\s+', '', m[0].lower()) + ":", lbl)

# Clean OCR/text artifacts
def clean(text: str) -> str:
    t = text.replace('\u00ad', '')
    t = re.sub(r'(?<=\w)-\n(?=\w)', "", t)
    return re.sub(r'\s*\n\s*', " ", t).strip()

# Convert a fitz Pixmap to PIL
def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

def ocr_image(im: Image.Image) -> str:
    return clean(pytesseract.image_to_string(im, lang="eng"))

# ─── 2. Paths & Gemini setup ────────────────────────────────────────
PDF_DIR     = Path(input("📂  Folder with your PDFs ➜ ").strip()).expanduser()
OUT_IMG_DIR = PDF_DIR / "_pics"
DB_PATH     = PDF_DIR / "chunks.db"
STORE_DIR   = PDF_DIR / "chroma_store"

# Clean previous runs
if DB_PATH.exists():
    DB_PATH.unlink()
if STORE_DIR.exists():
    shutil.rmtree(STORE_DIR)
shutil.rmtree(OUT_IMG_DIR, ignore_errors=True)
OUT_IMG_DIR.mkdir(exist_ok=True)

GEM_KEY = getpass.getpass("🔑  Gemini API key ➜ ").strip()
genai.configure(api_key=GEM_KEY)
GEM_MM = genai.GenerativeModel("gemini-1.5-flash-latest")

# ─── 3. SQLite registry ─────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.create_function("REGEXP", 2, lambda expr, itm: 1 if itm and re.search(expr, itm) else 0)
cur = conn.cursor()
cur.execute("""
CREATE TABLE chunks(
  chunk_id        INTEGER PRIMARY KEY,
  source          TEXT,
  page            INTEGER,
  type            TEXT,
  content         TEXT,
  caption         TEXT,
  img_path        TEXT,
  parent_chunk_id INTEGER,
  label_key       TEXT
)
""")
conn.commit()

# ─── 4. Embedding setup ─────────────────────────────────────────────
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
embedder   = HuggingFaceEmbeddings(model_name=EMB_MODEL)

# ─── 5. Convert PDFs via Docling ────────────────────────────────────
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_options=PdfPipelineOptions(
            images_scale=2.0,
            generate_picture_images=True
        )
    )
})
docling_docs = {}
for pdf in PDF_DIR.glob("*.pdf"):
    print("→ parsing", pdf.name)
    docling_docs[pdf.name] = converter.convert(str(pdf)).document

# ─── 6. Chunking & Extraction ───────────────────────────────────────
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024, stride=200)
chunks_by_page = defaultdict(lambda: defaultdict(list))
cid = 0

for src, doc in docling_docs.items():
    stem    = Path(src).stem.lower()
    img_dir = OUT_IMG_DIR / stem
    img_dir.mkdir(exist_ok=True)

    # 6.1 Text & Equations
    for ch in chunker.chunk(doc):
        page = ch.meta.doc_items[0].prov[0].page_no
        txt  = clean(ch.text)
        typ  = "equation" if len(txt) < 300 and re.search(r"(\\frac|\\sum|\\int|=|[∑∫√±×÷])", txt) else "text"
        if typ == "equation" and not re.search(r'\(\s*\d+\s*\)', txt):
            cid += 1
            txt = f"( {cid} ) {txt}"
        cid += 1
        cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, stem, page, typ, txt, None, None, None, None))

    # 6.2 OCR Fallback for Pages
    pdfdoc = fitz.open(str(PDF_DIR / src))
    for p in range(1, len(doc.pages) + 1):
        rows = cur.execute(
            "SELECT content FROM chunks WHERE source=? AND page=? AND type='text'",
            (stem, p)
        )
        if not any(rows):
            pix     = pdfdoc[p-1].get_pixmap(dpi=300)
            pil_img = pixmap_to_pil(pix)
            page_txt = clean(pytesseract.image_to_string(pil_img, lang="eng"))
            cid += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, stem, p, "page", page_txt, None, None, None, None))
    pdfdoc.close()

    # 6.3 Tables with Gemini captions
    tbl_no = 0
    for tbl in doc.tables:
        if not tbl.prov:
            continue
        tbl_no += 1
        page = tbl.prov[0].page_no
        md   = canon(tbl.export_to_markdown(doc))
        if not FIG_TBL.search(md):
            md = f"table{tbl_no}:\n{md}"
        prompt  = ["Write a one-sentence caption for this table:", md]
        summary = GEM_MM.generate_content(prompt).text.strip()
        cap     = f"table{tbl_no}: {summary}"
        lk      = label_key(cap)
        cid += 1
        cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, stem, page, "table", md, cap, None, None, lk))

    # 6.4 Images
    pdfdoc = fitz.open(str(PDF_DIR / src))
    fig_no = 0
    for i, pic in enumerate(doc.pictures, start=1):
        if not pic.prov:
            continue
        fig_no += 1
        page = pic.prov[0].page_no
        pil  = pic.get_image(doc)
        fp   = img_dir / f"fig{fig_no}_p{page}_{i}.png"
        pil.save(fp, "PNG")

        ocr_txt = ocr_image(pil)
        # placeholder caption, to be filled next
        cap     = None
        lk      = label_key(f"fig{fig_no}:")
        cid += 1
        cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                    (cid, stem, page, "image", ocr_txt, cap, str(fp), None, lk))
    pdfdoc.close()

    conn.commit()

# ─── 7. Summarize all images (so captions are never None) ──────────
def summarize_all_images():
    cur.execute("SELECT chunk_id, img_path FROM chunks WHERE type='image'")
    for cid, img_path in cur.fetchall():
        pil = Image.open(img_path)
        prompt  = ["Write a concise one-sentence summary of this figure:", pil]
        summary = GEM_MM.generate_content(prompt).text.strip()
        cur.execute("UPDATE chunks SET caption=? WHERE chunk_id=?", (summary, cid))
    conn.commit()
    print("✅ All image captions populated.")

summarize_all_images()  # populate captions before indexing

# ─── 8. Build Chroma vector store ──────────────────────────────────
docs = []
for cid, src, page, typ, txt, cap in cur.execute(
    "SELECT chunk_id, source, page, type, content, caption FROM chunks"
):
    if typ in ("image", "table"):
        # coalesce None->"" and prepend caption
        full = (cap or "") + "\n" + txt
    else:
        full = txt
    docs.append(Document(full, metadata={
        "chunk_id": cid,
        "source": src,
        "page": page,
        "type": typ
    }))

vectordb = Chroma(
    persist_directory=str(STORE_DIR),
    collection_name="multimodal_rag",
    embedding_function=embedder
)
vectordb.add_documents(docs)
vectordb.persist()

# ─── 9. Retrieval Helpers ─────────────────────────────────────────
def run_sim(q, k, filter=None):
    return vectordb.similarity_search(q, k, filter=filter)

def guess_src(q):
    stems = [Path(f).stem.lower() for f in docling_docs]
    tokens = re.findall(r'\w+', q.lower())
    for L in range(len(tokens), 0, -1):
        for i in range(len(tokens)-L+1):
            frag = "_".join(tokens[i:i+L])
            m = difflib.get_close_matches(frag, stems, n=1, cutoff=0.6)
            if m:
                return m[0]
    for w in set(tokens):
        for stem in stems:
            if w in stem:
                return stem
    hits = run_sim(q, 3, filter={"type":"page"})
    return hits[0].metadata["source"] if hits else None

def resolve_item(kind, key, src=None):
    m = re.fullmatch(r'([ivxlcdm\d]+)([a-z])?', key.lower())
    if m:
        raw, suff = m.groups()
        num = raw if raw.isdigit() else str(roman_to_int(raw))
        lk  = ('fig' if kind=='image' else 'table') + num
        if suff:
            lk += suff
        row = cur.execute(
            "SELECT chunk_id FROM chunks WHERE type=? AND label_key=? AND source=? ORDER BY page LIMIT 1",
            (kind, lk, src)
        ).fetchone()
        if row:
            return row[0]
    hits = run_sim(key, 5, filter={"type": kind})
    return hits[0].metadata["chunk_id"] if hits else None

def fetch(cid):
    return cur.execute(
        "SELECT source, page, type, content, caption, img_path FROM chunks WHERE chunk_id=?",
        (cid,)
    ).fetchone()

def show_single(cid, kind):
    if cid is None:
        print(f"No matching {kind}.")
        return
    src, p, typ, cont, cap, ip = fetch(cid)
    print(f"\n📄 {src} | p.{p}\n")
    if kind == "image":
        print(f"Caption: {cap}\n")
        display(Image.open(ip))
    elif kind == "table":
        print(f"{cap}\n")
        display(Markdown(cont))
    else:
        print(cont + "\n")

def list_sources():
    print("Available PDFs:")
    for i, name in enumerate(docling_docs, 1):
        print(f"  {i}. {name}")

# ─── 10. Q&A & Chat Loop ───────────────────────────────────────────
def answer(q: str) -> str:
    hits = run_sim(q, 5, filter={"type":"text"})
    context = "\n\n".join(h.page_content for h in hits)
    prompt = [
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    ]
    return GEM_MM.generate_content(prompt).text.strip()

def ask(q: str):
    ql = q.lower().strip()
    if ql in ("list sources", "sources", "list files", "files"):
        return list_sources()
    if ql in ("show summary of all images", "summaries of all images"):
        return summarize_all_images()
    if ql.startswith("find image by summary"):
        desc = ql.replace("find image by summary", "").strip()
        hits = run_sim(desc, 5, filter={"type":"image"})
        if not hits:
            print("No images found matching that description.")
        else:
            show_single(hits[0].metadata["chunk_id"], "image")
        return
    m = re.search(r'\bfig(?:ure)?\.?\s*([ivxlcdm\d]+[a-z]?)\b', ql)
    if m:
        src = guess_src(ql)
        cid = resolve_item('image', m.group(1), src)
        return show_single(cid, 'image')
    if "image" in ql:
        src = guess_src(ql)
        key = re.sub(r'.*image(?:\s+of)?\s+', '', ql)
        cid = resolve_item('image', key, None)
        hits = run_sim(key, 5, filter={"type":"image"})
        required = re.findall(r'\b(dct|psd|wt|wc)\b', key)
        for h in hits:
            cand = h.metadata["chunk_id"]
            _, _, _, ocr_txt, caption, _ = fetch(cand)
            text = (caption or "").lower() + " " + (ocr_txt or "").lower()
            if all(tok in text for tok in required):
                cid = cand
                break
        return show_single(cid or (hits[0].metadata["chunk_id"] if hits else None), 'image')
    m = re.search(r'\btable\s+([ivxlcdm\d]+[a-z]?)\b', ql)
    if m:
        src = guess_src(ql)
        cid = resolve_item('table', m.group(1), src)
        return show_single(cid, 'table')
    print("🤖 …thinking…")
    print(answer(q), "\n")

def chat():
    print("\n💬  Multimodal RAG • Text, Table & Image Retrieval\n")
    while True:
        q = input("🟢 You: ").strip()
        if not q:
            break
        ask(q)

# ─── 11. Entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    chat()
