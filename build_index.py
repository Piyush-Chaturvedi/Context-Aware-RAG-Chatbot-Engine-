"""
Run this once (and again any time documents are added/changed) to build
the searchable indexes for both the customer-support mode and the
internal engineering mode.

Usage:
    python build_index.py
"""
import os
from src.vectorstore import VectorStore
from src.config import CUSTOMER_DOCS_DIR, INTERNAL_DOCS_DIR, INDEX_DIR


def main():
    print("Building customer support index...")
    customer_store = VectorStore().build_index(CUSTOMER_DOCS_DIR)
    customer_store.save(os.path.join(INDEX_DIR, "customer_index.pkl"))
    print(f"  -> {len(customer_store.chunks)} chunks indexed from {CUSTOMER_DOCS_DIR}")

    print("Building internal engineering index...")
    internal_store = VectorStore().build_index(INTERNAL_DOCS_DIR)
    internal_store.save(os.path.join(INDEX_DIR, "internal_index.pkl"))
    print(f"  -> {len(internal_store.chunks)} chunks indexed from {INTERNAL_DOCS_DIR}")

    print("\nDone. Indexes saved in the 'index/' folder.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
