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

    def load_all(self) -> List[Document]:
        all_docs = []
        files = (
            list(self.data_dir.glob("*.txt")) +
            list(self.data_dir.glob("*.pdf"))
        )
        if not files:
            logger.warning(f"No files found in {self.data_dir}")
            return []
        for f in files:
            docs = self.load_txt(f)
            all_docs.extend(docs)
            logger.info(f"Loaded: {f.name}")
        logger.info(f"Total docs: {len(all_docs)}")
        return all_docs

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i
            c.metadata["chunk_len"] = len(c.page_content)
        logger.info(f"Total chunks: {len(chunks)}")
        return chunks

    def load_and_chunk(self) -> List[Document]:
        docs = self.load_all()
        if not docs:
            return []
        return self.chunk(docs)