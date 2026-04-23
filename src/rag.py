from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer


KB_PATH = Path("data/autostream_kb.md")


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def build_retriever(k: int = 3):
    text = KB_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    docs = splitter.create_documents([text])

    embeddings = SentenceTransformerEmbeddings()
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})