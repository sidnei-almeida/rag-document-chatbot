import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = logging.getLogger("docmind")
NOME_ARQUIVO_PDF = "documento.pdf"


def main():
    setup_logging()

    if not os.path.exists(NOME_ARQUIVO_PDF):
        logger.error("Arquivo '%s' não encontrado na pasta.", NOME_ARQUIVO_PDF)
        return

    logger.info("Lendo: %s", NOME_ARQUIVO_PDF)
    loader = PyPDFLoader(NOME_ARQUIVO_PDF)
    documentos = loader.load()
    logger.info("PDF carregado: %s páginas", len(documentos))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=list(settings.TEXT_SPLITTER_SEPARATORS),
    )
    textos_divididos = text_splitter.split_documents(documentos)
    logger.info("Documento dividido em %s chunks", len(textos_divididos))

    logger.info("Carregando embeddings: %s", settings.EMBEDDING_MODEL_NAME)
    embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    logger.info("Criando índice FAISS...")
    vector_store = FAISS.from_documents(textos_divididos, embeddings_model)
    vector_store.save_local(settings.VECTOR_STORE_PATH)
    logger.info("Índice salvo em '%s'", settings.VECTOR_STORE_PATH)


if __name__ == "__main__":
    main()
