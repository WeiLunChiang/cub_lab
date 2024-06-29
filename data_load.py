# %%
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from proj_prompt import rag_prompt
from langchain_core.runnables import RunnableLambda

# %%
import os
from collections import Counter
from langchain_core.runnables import RunnableLambda

# %%
# import chromadb
# from langchain_chroma import Chroma
from utils_vectordb import upsert_docs_to_vectordb, clear_collection
from langchain_core.documents import Document

# %%
doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})
doc3 = Document(page_content="doggy2", metadata={"source": "doggy.txt"})
doc4 = Document(page_content="doggy3", metadata={"source": "doggy.txt"})
# %%

upsert_docs_to_vectordb([doc1, doc2, doc3, doc4], cleanup="incremental")
# %%
from utils_vectordb import get_or_create_chromadb, get_metadata_runnable

# from functools import partial
# from collections import Counter
# from langchain_core.runnables import RunnableLambda
# def _get_metadata(key, x):
#     lst = [ix.metadata[key] for ix in x]
#     if len(lst) == 1:
#         return lst[0]
#     else:
#         counter = Counter(lst)
#         most_common_element = counter.most_common(1)[0][0]
#         return most_common_element

# def get_metadata_runnable(meta_data_key):
#     f = partial(_get_metadata, meta_data_key)
#     return RunnableLambda(f)


vs = get_or_create_chromadb("./chromadb", "collect1")
retriever = vs.as_retriever(search_kwargs={"k": 5})

chain = retriever | get_metadata_runnable(metadata_key="source")

# %%
response = chain.invoke("dog")
response


# %%
retriever.invoke("dog")[0].page_content

# %%
clear_collection(
    db_path="./chromadb",
    collection_name="collect1",
    source_id_key="source",
)

# %%
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = CSVLoader(file_path="./iris.csv")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
splits = text_splitter.split_documents(docs)


upsert_docs_to_vectordb(
    docs,
    db_path="./chromadb",
    collection_name="collect1",
    cleanup="incremental",
)

# %%
from utils_vectordb import get_or_create_chromadb, get_or_create_chroma_collection

vector_db = get_or_create_chromadb("./chromadb", "collect1")
collection = get_or_create_chroma_collection("./chromadb", "collect1")


# %%
def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectordb, cleanup="full", source_id_key="source")


# %%

_clear()
# %%
index(
    [doc1, doc2],
    record_manager,
    vectordb,
    cleanup=None,
    source_id_key="source",
)
# %%
index(
    [doc5],
    record_manager,
    vectordb,
    cleanup="incremental",
    source_id_key="source",
)

# %%


# %%
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# %%
retriever.invoke("sentosa")
# %%
# retrieverdocs

# %%
len(emb[0])


# %%
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os

path = "./chromadb"
client = chromadb.PersistentClient(path=path)
collection = client.get_or_create_collection(
    name="collect01", embedding_function=openai_ef
)

# %%
collection.peek()
# %%
openai_ef("hi")
# %%


from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader

loader = PyMuPDFLoader("./docs.pdf")
pages = loader.load_and_split()
# %%
pages[0].metadata
# %%
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("./TSMC 2023Q4 Consolidated Financial Statements_C.pdf")
pages = loader.load_and_split()
# %%
pages[0].metadata["123"] = 1234
# %%
pages[0].metadata
# %%
len(pages)
# %%
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# %%

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)
# %%
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)

# This example only specifies a relevant query
retriever.invoke("What are two movies about dinosaurs")
# %%
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser
# %%
print(prompt.format(query="請給我1995年的內容"))
# %%
import chromadb
from utils_vectordb import get_or_create_http_chromadb, clear_collection


vectordb = get_or_create_http_chromadb(
    host="localhost",
    port=8000,
    collection_name="collect01",
)
# %%
collect01 = chroma_client.get_or_create_collection(name="collect01")
# %%
clear_collection()
# %%
from utils_vectordb import get_or_create_http_chromadb, get_metadata_runnable

vectorstore = get_or_create_http_chromadb(
    host="localhost",
    port=8000,
    collection_name="collect01",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

chain = retriever | get_metadata_runnable(metadata_key="source")
chain.invoke("foo")
# %%
