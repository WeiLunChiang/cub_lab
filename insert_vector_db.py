# %%
import itertools
import os
import pickle
import time
import pandas as pd
from tqdm import tqdm

# from langchain_community.document_loaders.csv_loader import CSVLoader
from utils_vectordb import CustomCSVLoader as CSVLoader
from utils_vectordb import get_or_create_http_chromadb

# %%


def create_csv():

    qa = pd.read_excel("./cubelab.xlsx", sheet_name="QA")
    data_time = pd.read_excel("./cubelab.xlsx", sheet_name="時間")
    data_name = pd.read_excel("./cubelab.xlsx", sheet_name="特店名稱")
    data_category = pd.read_excel("./cubelab.xlsx", sheet_name="消費類別")

    # %%
    list_category = list(data_category["消費類別"])
    list_name = list(data_name["特店名稱"])
    list_time = list(data_time["替代變數"])

    # %%
    l = []

    # %%
    for t in list_time:
        d = qa[qa["編號"] == "A"]["變形問題"].apply(
            lambda x: x.replace("[某個消費時間]", str(t))
        )
        l.append(
            pd.DataFrame(
                {
                    "category": "A",
                    "Q": d,
                }
            )
        )
    # %%
    for t, n in itertools.product(list_time, list_name):

        d = qa[qa["編號"] == "B"]["變形問題"].apply(
            lambda x: x.replace("[某個消費時間]", str(t)).replace("[某個商戶]", str(n))
        )
        l.append(
            pd.DataFrame(
                {
                    "category": "B",
                    "Q": d,
                }
            )
        )
    # %%
    for t, c in itertools.product(list_time, list_category):

        d = qa[qa["編號"] == "C"]["變形問題"].apply(
            lambda x: x.replace("[某個消費時間]", str(t)).replace("[消費類別]", str(c))
        )
        l.append(
            pd.DataFrame(
                {
                    "category": "C",
                    "Q": d,
                }
            )
        )
    # %%
    for t in list_time:
        d = qa[qa["編號"] == "D"]["變形問題"].apply(
            lambda x: x.replace("[時間]", str(t))
        )
        l.append(
            pd.DataFrame(
                {
                    "category": "D",
                    "Q": d,
                }
            )
        )
    # %%
    df = pd.concat(l, axis=0)
    df.reset_index(drop=True, inplace=True)
    a_df = (
        qa[["編號", "問題類別", "SQL1", "SQL2", "SQL3"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    a_df.rename({"編號": "category"}, axis=1, inplace=True)
    # %%
    a_df["SQL1"] = a_df["SQL1"].apply(lambda x: x.replace("\n", " "))
    a_df["SQL2"] = a_df["SQL2"].apply(lambda x: x.replace("\n", " "))
    a_df["SQL3"] = a_df["SQL3"].apply(lambda x: x.replace("\n", " "))

    # %%
    df = df.merge(a_df, on="category", how="left")

    # %%
    df.rename({"Q": "變形問題"}, axis=1, inplace=True)
    # %%
    df.to_csv("qa_set_with_sql.csv", index=False)


def insert_to_vector_db(
    num_batches=1,
    max_retries=20,
    file_path="qa_set_with_sql_lite.csv",
    collection_name="collect_cubelab_qa_lite",
    metadata_columns=["category", "變形問題", "SQL1", "SQL2", "SQL3"],
    source_column="問題類別",
):
    # 載入CSV檔案，並指定metadata欄位
    loader = CSVLoader(
        file_path=file_path,
        metadata_columns=metadata_columns,
        source_column=source_column,
    )
    docs = loader.load()
    # breakpoint()
    list_length = len(docs)
    num_batches = num_batches
    batch_size = list_length // num_batches
    vector_db = get_or_create_http_chromadb(
        collection_name=collection_name,
        emb_type="openai",
    )

    max_retries = max_retries
    resume_file = "resume.pkl"

    # 如果resume_file存在，則讀取c的值，否則將c設置為初始值0
    if os.path.exists(resume_file):
        with open(resume_file, "rb") as f:
            c = pickle.load(f)
    else:
        c = 0

    # 進行資料批次上傳
    for _ in range(max_retries):
        try:
            for i in tqdm(range(c, num_batches)):
                c = i
                batch = list(
                    itertools.islice(docs, i * batch_size, (i + 1) * batch_size)
                )
                # breakpoint()
                vector_db.add_documents(batch)
                # 每次成功處理一個batch後，將c的值寫入resume_file
                with open(resume_file, "wb") as f:
                    pickle.dump(c, f)
                break
            # 完成所有批次上傳後，刪除resume_file
            if os.path.exists(resume_file):
                os.remove(resume_file)
            break  # 如果操作成功，則跳出迴圈
        except Exception as e:
            print(f"錯誤: {e}, 正在重試 ({_+1}/{max_retries})")
            time.sleep(60)
    else:
        print("已達到最大重試次數，操作失敗")


def create_csv_ner():

    qa = pd.read_excel("./cubelab.xlsx", sheet_name="QA")
    qa.rename(
        {
            "編號": "category",
            "Q": "變形問題",
        },
        axis=1,
        inplace=True,
    )
    qa.to_csv("qa_set_with_sql_lite.csv", index=False)
    print(qa.columns)


if __name__ == "__main__":
    create_csv_ner()
    insert_to_vector_db(
        num_batches=1,
        max_retries=20,
        file_path="qa_set_with_sql_lite.csv",
        collection_name="collect_cubelab_qa_lite_2",
        metadata_columns=["category", "問題類別", "SQL1", "SQL2", "SQL3"],
        source_column="變形問題"
    )


# %%

# from utils_vectordb import CustomCSVLoader
# loader = CustomCSVLoader(
#     file_path="qa_set_with_sql.csv",
#     metadata_columns=["category", "問題類別", "SQL1", "SQL2", "SQL3"],
#     source_column='變形問題',
# )
# docs = loader.load()
# docs[0]
# %%
