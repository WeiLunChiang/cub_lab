from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils_vectordb import (
    get_or_create_http_chromadb,
    upsert_docs_to_vectordb,
    get_or_create_chroma_collection,
    get_or_create_chroma_http_collection,
)
from time import sleep
from tqdm import tqdm
import itertools
import time


# %%
loader = CSVLoader(
    file_path="qa_set_with_sql.csv",
    metadata_columns=["category", "問題類別", "SQL1", "SQL2", "SQL3"],
)

# for i in range(num_batches):
#     batch = list(itertools.islice(docs, i*batch_size, (i+1)*batch_size))
#     print(i)
#     vector_db.add_documents(batch)
# upsert_docs_to_vectordb(
#     batch,
#     host="127.0.0.1",
#     port="8000",
#     collection_name="collect01",
#     cleanup="incremental",
#     batch_size=1000,
# )

docs = loader.load()
list_length = len(docs)
num_batches = 150
batch_size = list_length // num_batches
vector_db = get_or_create_http_chromadb(
    collection_name="collect_cubelab_qa_00",
    emb_type="openai",
)


max_retries = 20
c = 121
for _ in range(max_retries):
    try:
        for i in range(c, num_batches):
            print(i)
            c = i
            batch = list(itertools.islice(docs, i * batch_size, (i + 1) * batch_size))
            vector_db.add_documents(batch)
        break  # 如果操作成功，則跳出迴圈
    except Exception as e:
        print(f"錯誤: {e}, 正在重試 ({_+1}/{max_retries})")
        time.sleep(60)
else:
    print("已達到最大重試次數，操作失敗")
