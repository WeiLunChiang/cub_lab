from langchain_community.document_loaders.csv_loader import CSVLoader
from tqdm import tqdm
import itertools
import os
import pickle
import time

from utils_vectordb import (
    get_or_create_http_chromadb,
)

# 載入CSV檔案，並指定metadata欄位
loader = CSVLoader(
    file_path="qa_set_with_sql.csv",
    metadata_columns=["category", "問題類別", "SQL1", "SQL2", "SQL3"],
)

docs = loader.load()
list_length = len(docs)
num_batches = 150
batch_size = list_length // num_batches
vector_db = get_or_create_http_chromadb(
    collection_name="collect_cubelab_qa_00",
    emb_type="openai",
)

max_retries = 20
resume_file = 'resume.pkl'

# 如果resume_file存在，則讀取c的值，否則將c設置為初始值0
if os.path.exists(resume_file):
    with open(resume_file, 'rb') as f:
        c = pickle.load(f)
else:
    c = 0

# 進行資料批次上傳
for _ in range(max_retries):
    try:
        for i in tqdm(range(c, num_batches)):
            c = i
            batch = list(itertools.islice(docs, i * batch_size, (i + 1) * batch_size))
            vector_db.add_documents(batch)
            # 每次成功處理一個batch後，將c的值寫入resume_file
            with open(resume_file, 'wb') as f:
                pickle.dump(c, f)
        # 完成所有批次上傳後，刪除resume_file
        if os.path.exists(resume_file):
            os.remove(resume_file)
        break  # 如果操作成功，則跳出迴圈
    except Exception as e:
        print(f"錯誤: {e}, 正在重試 ({_+1}/{max_retries})")
        time.sleep(60)
else:
    print("已達到最大重試次數，操作失敗")
