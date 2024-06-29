import os
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index


def _get_Embeddings_func(emb_type: str = "azure"):
    """
    搭配chroma ,原生api無法操作
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

def _get_azure_openai_ef():
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
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=_get_azure_openai_ef()
    )
    return collection


def _get_or_create_chroma_record_manager(
    collection_name, db_url="sqlite:///record_manager_cache.db"
):
    record_manager = SQLRecordManager(
        namespace=f"chroma/{collection_name}",
        db_url=db_url,
    )
    record_manager.create_schema()
    return record_manager


def upsert_docs_to_vectordb(
    docs,
    host: str = "127.0.0.1",
    port: str = "8000",
    collection_name: str = "collect01",
    source_id_key="source",
    batch_size: int = 1000,
    cleanup=None,
) -> None:

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
    source_id_key="source",
) -> None:

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


from functools import partial
from collections import Counter
from langchain_core.runnables import RunnableLambda


def _get_metadata(key, x):
    lst = [ix.metadata[key] for ix in x]
    if len(lst) == 1:
        return lst[0]
    else:
        counter = Counter(lst)
        most_common_element = counter.most_common(1)[0][0]
        return most_common_element


def get_metadata_runnable(metadata_key):
    f = partial(_get_metadata, metadata_key)
    return RunnableLambda(f)
