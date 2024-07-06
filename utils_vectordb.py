import os
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index

from functools import partial
from collections import Counter
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from typing import List, Dict, Any, Union


class CustomCSVLoader(CSVLoader):
    """
    Custom CSV Loader that modifies page content by removing a
    specified prefix.
    """

    def __init__(self, file_path: str, metadata_columns: List[str], source_column: str):
        """
        Initialize with file path, metadata columns, and source
        column.

        :param file_path: Path to the CSV file.
        :param metadata_columns: List of metadata column names.
        :param source_column: Name of the source column.
        """
        super().__init__(
            file_path=file_path,
            metadata_columns=metadata_columns,
            source_column=source_column,
        )
        self.file_name = os.path.basename(file_path)
        self.source_column = source_column

    def load(self) -> List[Document]:
        """
        Load and process CSV file, removing prefix from page content.

        :return: List of processed Document objects.
        """
        docs = super().load()
        prefix = f"{self.source_column}: "
        for doc in docs:
            if doc.page_content.startswith(prefix):
                doc.page_content = doc.page_content[len(prefix) :]
            doc.metadata["source"] = self.file_name
        return docs


def _get_Embeddings_func(
    emb_type: str = "azure",
) -> Union[AzureOpenAIEmbeddings, OpenAIEmbeddings]:
    """
    Get the embeddings function based on the specified type.

    :param emb_type: Type of embeddings to use ("azure" or other).
    :return: Embeddings function.
    """
    if emb_type == "azure":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


from chromadb.utils import embedding_functions


def _get_azure_openai_ef() -> embedding_functions.OpenAIEmbeddingFunction:
    """
    Get Azure OpenAI embedding function.

    :return: Azure OpenAI embedding function.
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_type="azure",
        api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        deployment_id=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    )
    return openai_ef


def get_or_create_chromadb(
    path: str,
    collection_name: str,
    emb_type: str = "azure",
) -> Chroma:
    """
    Get or create a ChromaDB instance with the specified path and
    collection name.

    :param path: Path to the ChromaDB storage.
    :param collection_name: Name of the collection.
    :param emb_type: Type of embeddings to use ("azure" or other).
    :return: ChromaDB instance.
    """
    if os.path.exists(path):
        print(f"reload vectordb from {path}, collection name={collection_name}")
        vectordb = Chroma(
            client=chromadb.PersistentClient(path),
            collection_name=collection_name,
            embedding_function=_get_Embeddings_func(emb_type=emb_type),
        )
    else:
        print(f"create vectordb at {path}, collection name={collection_name}")
        vectordb = Chroma(
            client=chromadb.PersistentClient(path),
            collection_name=collection_name,
            embedding_function=_get_Embeddings_func(emb_type=emb_type),
        )
    return vectordb


def get_or_create_http_chromadb(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
    emb_type: str = "azure",
) -> Chroma:
    """
    Get or create a ChromaDB instance via HTTP with the specified
    host, port, and collection name.

    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :param emb_type: Type of embeddings to use ("azure" or other).
    :return: ChromaDB instance.
    """
    client = chromadb.HttpClient(
        host=host,
        port=port,
    )
    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_get_Embeddings_func(emb_type=emb_type),
    )
    return vectordb


def get_or_create_chroma_http_collection(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
) -> "chroma_collection":
    """
    Get or create a ChromaDB collection via HTTP with the specified
    host and port.

    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :return: ChromaDB collection.
    """
    client = chromadb.HttpClient(
        host=host,
        port=port,
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_azure_openai_ef(),
    )
    return collection


def get_or_create_chroma_collection(
    chromadb_path: str = "./chromadb",
    collection_name: str = "collect01",
) -> "chroma_collection":
    """
    Get or create a ChromaDB collection with the specified path and
    collection name.

    :param chromadb_path: Path to the ChromaDB storage.
    :param collection_name: Name of the collection.
    :return: ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=_get_azure_openai_ef()
    )
    return collection


def _get_or_create_chroma_record_manager(
    collection_name: str, db_url: str = "sqlite:///record_manager_cache.db"
) -> SQLRecordManager:
    """
    Get or create a Chroma record manager with the specified
    collection name and database URL.

    :param collection_name: Name of the collection.
    :param db_url: Database URL for the record manager.
    :return: Chroma record manager.
    """
    record_manager = SQLRecordManager(
        namespace=f"chroma/{collection_name}",
        db_url=db_url,
    )
    record_manager.create_schema()
    return record_manager


def upsert_docs_to_vectordb(
    docs: List[Document],
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
    source_id_key: str = "source",
    batch_size: int = 1000,
    cleanup: Union[None, str] = None,
) -> None:
    """
    Upsert documents to ChromaDB.

    :param docs: List of documents to upsert.
    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :param source_id_key: Key for the source ID.
    :param batch_size: Batch size for upserting documents.
    :param cleanup: Cleanup mode.
    """
    vectordb = get_or_create_http_chromadb(
        host=host,
        port=port,
        collection_name=collection_name,
        embedding_function=_get_Embeddings_func(),
    )
    record_manager = _get_or_create_chroma_record_manager(collection_name)

    result = index(
        docs,
        record_manager,
        vectordb,
        cleanup=cleanup,
        source_id_key=source_id_key,
        batch_size=batch_size,
    )
    print(
        f"upsert_docs_to_vectordb {vectordb}/{collection_name} in {cleanup} mode, \n{result}"
    )


def clear_collection(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
    source_id_key: str = "source",
) -> None:
    """
    Clear a ChromaDB collection.

    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :param source_id_key: Key for the source ID.
    """
    vectordb = get_or_create_http_chromadb(
        host=host,
        port=port,
        collection_name=collection_name,
        embedding_function=_get_Embeddings_func(),
    )
    record_manager = _get_or_create_chroma_record_manager(collection_name)
    result = index(
        [], record_manager, vectordb, cleanup="full", source_id_key=collection_name
    )
    print(f"clear {vectordb}/ {collection_name}, \n{result}")


def _get_metadata(key: str, x: List[Document]) -> Any:
    """
    Get the most common metadata value for the specified key.

    :param key: Metadata key to look for.
    :param x: List of documents to search.
    :return: Most common metadata value.
    """
    lst = [ix.metadata[key] for ix in x]
    if len(lst) == 1:
        return lst[0]
    else:
        counter = Counter(lst)
        most_common_element = counter.most_common(1)[0][0]
        return most_common_element


def get_metadata_runnable(metadata_key: str) -> RunnableLambda:
    """
    Get a runnable to fetch metadata by key.

    :param metadata_key: Metadata key to fetch.
    :return: RunnableLambda function to get metadata.
    """
    f = partial(_get_metadata, metadata_key)
    return RunnableLambda(f)


def delete_data_from_chroma(
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect_cubelab_qa_test999",
    filter_criteria: Dict[str, Any] = {"category": "example_category"},
) -> None:
    """
    Delete data from a ChromaDB collection based on filter criteria.

    :param host: Host for the ChromaDB HTTP server.
    :param port: Port for the ChromaDB HTTP server.
    :param collection_name: Name of the collection.
    :param filter_criteria: Criteria to filter documents to delete.
    """
    collection = get_or_create_chroma_http_collection(
        host=host,
        port=port,
        collection_name=collection_name,
    )
    documents_to_delete = collection.get(where=filter_criteria)
    collection.delete(ids=documents_to_delete["ids"])

    print(
        f"collection: {collection_name} 資料{len(documents_to_delete['ids'])}筆 刪除完成"
    )


if __name__ == "__main__":
    filter_criteria = {"category": "A"}
    delete_data_from_chroma(
        collection_name="collect_cubelab_qa_test999", filter_criteria=filter_criteria
    )
