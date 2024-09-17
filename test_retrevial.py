# %%
def create_json(**kwargs):
    import os
    import math
    import pickle
    import json
    from tqdm import tqdm

    from langchain_core.runnables import RunnableLambda
    from utils_retriever import (
        RetrieveWithScore,
        get_metadata_runnable,
        get_page_content_runnable,
    )
    from utils_vectordb import get_or_create_http_chromadb

    K = kwargs.get("k", 1)

    # 輸入檔案
    input_file = "test/batch_all.json"
    # 輸出檔案
    output_file = "test/output_retreival.json"
    # 進度檔案
    progress_file = "batch_progress.pkl"
    batch_size = 50

    vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_lite")

    retriever = RetrieveWithScore(vectorstore, k=K, score_threshold=0)

    retriever_chain = (
        RunnableLambda(lambda response: response["response"]["modify_query"])
        | retriever
        | {  # vectordb拉到的內容(包含SQL)
            "標準問題": get_metadata_runnable("問題類別"),
            "page_content": get_page_content_runnable(),
            "category": get_metadata_runnable("category"),
            "score": get_metadata_runnable("score"),
        }
    )

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 嘗試讀取進度檔案，若不存在則從頭開始
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            start_batch = pickle.load(f)
    else:
        start_batch = 0

    # 計算總批次數
    total_batches = math.ceil(len(data) / batch_size)

    # 批次處理
    for i in tqdm(range(start_batch, total_batches)):
        # 獲取當前批次的數據
        batch_data = data[i * batch_size : (i + 1) * batch_size]

        # 調用 retriever_chain.batch() 執行當前批次
        batch_results = retriever_chain.batch(batch_data)

        l = []
        for r, d in zip(batch_results, batch_data):
            r["問題類別"] = d["category"]
            r["變形問題"] = d["question"]
            r["變形問題_挖空"] = d["response"]["modify_query"]
            l.append(r)

        # 將批次結果保存為獨立的 JSON 文件
        batch_output_file = f"batch_results_{i + 1}.json"
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(l, f, ensure_ascii=False, indent=4)

        # 保存當前進度
        with open(progress_file, "wb") as f:
            pickle.dump(i + 1, f)  # 保存下一個要處理的批次索引

    # 合併所有中繼檔案
    all_results = []
    for i in range(start_batch, total_batches):
        batch_output_file = f"batch_results_{i + 1}.json"
        if os.path.exists(batch_output_file):
            with open(batch_output_file, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
            # 刪除中繼檔案
            os.remove(batch_output_file)

    # 將所有結果寫入總輸出檔案
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 刪除進度檔案（處理完成後）
    if os.path.exists(progress_file):
        os.remove(progress_file)

    # 日誌輸出結果數量
    print(f"Total results saved: {len(all_results)}")


# TODO: 加入page_content欄位


def summary_json(**kwargs):
    # %%
    import json
    import numpy
    import pandas as pd

    K = kwargs.get("k", 1)

    JSON_FILE_PATH = output_file = "test/output_retreival.json"
    EXP_OUTPUT_PATH = "exp_output/正確率計算_v5_t00.xlsx"
    FINAL_OUTPUT_PATH = f"exp_output/vdb_k{K}_正確率計算_v5_t00.xlsx"

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    # %%

    df = pd.DataFrame(
        results,
        columns=[
            "問題類別",
            "變形問題",
            "標準問題",
            "category",
            "page_content",
        ],
    )

    # 因為'標準問題'和'category','page_content'的值是列表，所以我們需要解開這些列表
    df["標準問題"] = df["標準問題"].apply(lambda x: x[0])
    df["category"] = df["category"].apply(lambda x: x[0])
    df["page_content"] = df["page_content"].apply(lambda x: x[0])
    df["_category_q"] = (df.category == df.問題類別).astype("int")
    vdb_result = df[
        df.問題類別.isin(
            [
                "A",
                "B",
                "C",
                "D",
            ]
        )
    ]

    vdb_result.rename({"category": "VDB類別"}, axis=1, inplace=True)
    test_qa_result = pd.read_excel(EXP_OUTPUT_PATH)
    test_qa_result_withvdb = test_qa_result.merge(
        vdb_result[["變形問題", "VDB類別", "page_content", "_category_q", "標準問題"]],
        on="變形問題",
        how="inner",
    )

    test_qa_result_withvdb.to_excel(FINAL_OUTPUT_PATH)


if __name__ == "__main__":
    for k in [1, 3, 5]:
        print(f"k = {k}")
        create_json(k=k)
        summary_json(k=k)
        print("=" * 50)

# %%
