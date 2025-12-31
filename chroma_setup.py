import os
import shutil
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_chroma_db(chroma_path, file_paths, embedding_model_name):
    """
    Sets up the Chroma database with the given files and embedding model.

    Args:
        chroma_path (str): Path to persist the Chroma database.
        file_paths (list): List of file paths to load documents from.
        embedding_model_name (str): Name of the embedding model to use.

    Returns:
        Chroma: The initialized Chroma database.
    """
    load_dotenv()
    shutil.rmtree(chroma_path, ignore_errors=True)

    documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=chroma_path
    )
    db.persist()
    return db