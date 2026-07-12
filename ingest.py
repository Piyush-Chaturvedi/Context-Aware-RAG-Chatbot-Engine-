"""
Document ingestion: loads .txt / .pdf / .docx files from a folder,
cleans them, and splits them into overlapping word-chunks ready for embedding.
"""
import os
import re
from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def _read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required to read PDF files. Run: pip install pypdf")
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path):
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required to read .docx files. Run: pip install python-docx")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return _read_txt(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, source_name, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """
    Splits text into overlapping chunks of ~chunk_size words.
    Returns a list of dicts: {"text": ..., "source": ...}
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        if chunk_str.strip():
            chunks.append({"text": chunk_str, "source": source_name})
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def load_and_chunk_folder(folder_path):
    """
    Loads every supported file in a folder and returns a flat list of chunks
    with metadata pointing back to the source filename.
    """
    all_chunks = []
    if not os.path.isdir(folder_path):
        return all_chunks

    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".txt", ".pdf", ".docx"):
            continue
        try:
            raw = load_document(fpath)
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned, source_name=fname)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    return all_chunks
