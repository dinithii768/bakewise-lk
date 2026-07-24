"""
Document Loader for BakeWise LK
Loads .txt and .pdf files from data/raw/
Chunking Strategy:
  - RecursiveCharacterTextSplitter
  - chunk_size=700, chunk_overlap=120
  - Tries paragraph → sentence → word boundaries
"""

import logging
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import RAG_CONFIG, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:

    def __init__(self):
        self.data_dir = Path(RAW_DATA_DIR)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def load_txt(self, path: Path) -> List[Document]:
        try:
            text = path.read_text(encoding="utf-8")
            return [Document(
                page_content=text,
                metadata={"source": path.name, "type": "txt"}
            )]
        except Exception as e:
            logger.error(f"Error loading {path.name}: {e}")
            return []

    def load_pdf(self, path: Path) -> List[Document]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": i + 1, "type": "pdf"}
                    ))
            return pages
        except Exception as e:
            logger.error(f"Error loading PDF {path.name}: {e}")
            return []

    def load_all(self) -> List[Document]:
        all_docs = []
        files = list(self.data_dir.glob("*.txt")) + list(self.data_dir.glob("*.pdf"))

        if not files:
            logger.warning(f"No files found in {self.data_dir}")
            return []

        for f in files:
            if f.suffix == ".txt":
                docs = self.load_txt(f)
            elif f.suffix == ".pdf":
                docs = self.load_pdf(f)
            else:
                continue
            all_docs.extend(docs)
            logger.info(f"Loaded: {f.name}")

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_len"] = len(chunk.page_content)
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def load_and_chunk(self) -> List[Document]:
        docs = self.load_all()
        if not docs:
            return []
        return self.chunk(docs)