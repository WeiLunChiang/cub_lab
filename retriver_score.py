# %%
import os
import json
import csv
import itertools
from typing import List, Callable, Any, Dict
from tqdm import tqdm
import pandas as pd
import time
from functools import wraps
from pathlib import Path
import chromadb
from utils_vectordb import get_or_create_chroma_collection


# %%
def query_db(
    queries: List[str],
    collection: Any = None,
    n_results: int = 50,
) -> List[List[float]]:
    """
    Query the database and return cosine similarities for given queries.

    Args:
        queries (List[str]): List of query texts.
        collection (Any): The collection to query from.
        n_results (int): Number of results to return.

    Returns:
        List[List[float]]: List of cosine similarities for each query.
    """

    results = collection.query(
        query_texts=queries, include=["documents", "distances"], n_results=n_results
    )
    all_distances = results["distances"]
    all_cosine_similarities = [
        [(1 - distance) for distance in distances] for distances in all_distances
    ]
    return all_cosine_similarities, results["documents"]


def save_results_to_json(filename: Path, results: List[Dict]):
    with filename.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def load_results_from_json(filename: Path) -> List[Dict]:
    if filename.exists():
        with filename.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(filename: Path, batch_idx: int):
    with filename.open("w", encoding="utf-8") as f:
        json.dump({"last_processed_batch": batch_idx}, f)


def load_progress(filename: Path) -> int:
    if filename.exists():
        with filename.open("r", encoding="utf-8") as f:
            progress = json.load(f)
            return progress.get("last_processed_batch", 0)
    return 0


def process_batches(data: pd.DataFrame, collection: Any, batch_size: int = 200):
    all_queries = list(data["變形問題"])
    all_categories = list(data["category"])

    num_batches = (len(all_queries) + batch_size - 1) // batch_size
    progress_filename = Path("./progress.json")

    start_batch = load_progress(progress_filename)
    with tqdm(
        total=len(all_queries),
        initial=start_batch * batch_size,
        unit="queries",
        desc="Processing",
        ncols=100,
        leave=True,
    ) as pbar:
        for batch_idx in range(start_batch, num_batches):
            batch_queries = all_queries[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_categories = all_categories[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]

            try:
                similarities, documents = query_db(batch_queries, collection)
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                continue

            results = []
            for j in range(len(batch_queries)):
                result = {
                    "問題類別": batch_categories[j],
                    "變形問題": batch_queries[j],
                    "相似度": similarities[j],
                    "document": documents[j],
                }
                results.append(result)

            batch_filename = Path(
                f"./json_files/query_results_batch_{batch_idx + 1}.json"
            )
            save_results_to_json(batch_filename, results)
            save_progress(progress_filename, batch_idx + 1)
            pbar.update(len(batch_queries))


def merge_data(json_dir: str, output_csv_file: str, fieldnames: List[str]) -> None:
    # 檢查JSON文件的目錄是否存在
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"The directory {json_dir} does not exist")

    # 搜集所有JSON文件的路徑
    json_files = [
        os.path.join(json_dir, filename)
        for filename in os.listdir(json_dir)
        if filename.endswith(".json")
    ]

    # 將合併的數據寫入一個CSV文件，並使用tqdm顯示進度條
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_path in tqdm(json_files, desc="Processing JSON files"):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    for row in data:
                        if "相似度" in row and isinstance(row["相似度"], list):
                            for similarity, document in zip(
                                row["相似度"], row["document"]
                            ):
                                new_row = row.copy()
                                new_row["相似度"] = similarity
                                new_row["document"] = document
                                writer.writerow(new_row)
                        else:
                            writer.writerow(row)
                else:
                    print(f"File {file_path} does not contain a list. Skipping.")


# %%
if __name__ == "__main__":
    client = chromadb.HttpClient(
        host="127.0.0.1",
        port="8000",
    )

    collection = get_or_create_chroma_collection(
        collection_name="collect_cubelab_qa_lite"
    )

    # data必要欄位: ["category","變形問題"]
    data = pd.read_csv("json_files_from_query_ner/ner_results_by_LLM.csv")
    data.rename({"modify_query": "變形問題"}, axis=1, inplace=True)
    process_batches(data, collection)

    merge_data(
        json_dir="json_files_from_query_ner",
        output_csv_file="json_files_from_query_ner/similarity_search_result/merged_data_with_extra_q.csv",
        fieldnames=["問題類別", "變形問題", "相似度", "document"],
    )

    df = pd.read_csv(
        "json_files_from_query_ner/similarity_search_result/merged_data_with_extra_q.csv"
    )

    df["排序"] = df.groupby(["問題類別", "變形問題"])["相似度"].rank(
        ascending=False,
        method="dense",
    )

    df1 = (
        df.groupby(["排序"])["相似度"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index(drop=False)
    )
    df2 = (
        df.groupby(["問題類別", "排序"])["相似度"]
        .agg(mean="mean", std="std", min="min", max="max")
        .reset_index(drop=False)
    )

    # %%
    def group_min_max(df, groups):
        result = pd.DataFrame()

        for group in groups:
            df_tmp = df[df.問題類別 == group]
            dmax = df_tmp.相似度.max()
            dmin = df_tmp.相似度.min()
            result = pd.concat(
                [result, (df_tmp[(df_tmp.相似度 == dmax) | (df_tmp.相似度 == dmin)])],
                axis=0,
            )
        result.reset_index(drop=True, inplace=True)

        return result

    # %%
    result_df = group_min_max(df, list("ABCDEFG"))

    # %%

    with pd.ExcelWriter(
        "json_files_from_query_ner/output.xlsx", engine="xlsxwriter"
    ) as writer:
        df1.to_excel(writer, sheet_name="相似度", index=False)
        df2.to_excel(writer, sheet_name="by 問題類別", index=False)
        result_df.to_excel(writer, sheet_name="相似度最大最小值", index=False)

    # %%
    if False:
        import pandas as pd

    # %%
    client = chromadb.HttpClient(
        host="127.0.0.1",
        port="8000",
    )
    collection = get_or_create_chroma_collection(collection_name="cubelab_02")

    financial_questions = [
        "我可以申請房屋貸款嗎？",
        "定期存款的利率是多少？",
        "如何開設投資帳戶？",
        "請問有沒有推薦的保險產品？",
        "如何進行基金定投？",
        "股票交易的手續費是多少？",
        "如何申請小額信貸？",
        "我能申請外幣存款帳戶嗎？",
        "買黃金有什麼投資策略？",
        "請問如何計算貸款的月供？",
        "教育儲蓄保險有哪些選擇？",
        "如何進行理財規劃？",
        "退休金計劃有什麼選擇？",
        "債券投資的風險有哪些？",
        "如何申請汽車貸款？",
        "健康保險的保障範圍有哪些？",
        "請問如何開始投資股票？",
        "外匯交易的基本知識是什麼？",
        "如何計算我的財務狀況？",
        "金融理財產品的風險等級如何區分？",
    ]

    # 清單2：詢問個人生活問題的問句
    life_questions = [
        "你今天過得怎麼樣？",
        "你最喜歡的興趣愛好是什麼？",
        "平時都喜歡做哪些運動？",
        "你有養寵物嗎？",
        "你最近在看什麼書？",
        "你對旅遊有什麼計劃？",
        "你喜歡做飯嗎？",
        "你最喜歡的電影是哪一部？",
        "你的生日是哪天？",
        "你住在哪個城市？",
        "你有兄弟姐妹嗎？",
        "你通常怎麼放鬆自己？",
        "你最喜歡的音樂類型是什麼？",
        "你最近在追哪些電視劇？",
        "你有什麼特別的技能嗎？",
        "你最喜歡的節日是什麼？",
        "你喜歡喝咖啡還是茶？",
        "你通常怎麼過週末？",
        "你對健康飲食有什麼看法？",
        "你有學過什麼樂器嗎？",
    ]

    # 平均 0.59
    # 清單3：詢問信用卡消費問題
    credit_card_questions = [
        "上個月我在蝦皮花了多少錢？",
        "我上次在全聯刷了多少錢？",
        "今年到目前為止我在加油站消費了多少？",
        "我上次在大潤發的消費金額是多少？",
        "最近一次在星巴克的消費是多少？",
        "我的信用卡這個月在餐廳消費了多少錢？",
        "上週我在家樂福的消費金額是多少？",
        "我在去年12月的總消費金額是多少？",
        "昨天我在誠品書店刷了多少錢？",
        "最近一次在便利商店的消費是多少？",
        "我上個月的娛樂支出總額是多少？",
        "我在上次電影院的消費是多少？",
        "最近一次在藥局的消費是多少？",
        "上個月我在網購上的總消費是多少？",
        "我最近一次在餐廳的消費是多少？",
        "我的信用卡上次繳了多少水電費？",
        "我上週的總消費金額是多少？",
        "最近一次在百貨公司的消費是多少？",
        "今年我在服飾店的總消費是多少？",
        "上個月我在外送平台花了多少錢？",
    ]
    range(-1, 1)
    l = [ix[0] for ix in query_db(financial_questions, collection, n_results=1)[0]]
    print(np.mean(l))
    l = [ix[0] for ix in query_db(life_questions, collection, n_results=1)[0]]
    print(np.mean(l))
    l = [ix[0] for ix in query_db(credit_card_questions, collection, n_results=1)[0]]
    print(np.mean(l))
# %%

# %%
