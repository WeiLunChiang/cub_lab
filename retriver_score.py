# %%
import json
import csv
from typing import List, Any, Dict
from tqdm import tqdm
import pandas as pd
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
    progress_filename = Path(
        "json_files_from_query_ner/similarity_search_result/progress.json"
    )

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
                f"json_files_from_query_ner/similarity_search_result/query_results_batch_{batch_idx + 1}.json"
            )
            save_results_to_json(batch_filename, results)
            save_progress(progress_filename, batch_idx + 1)
            pbar.update(len(batch_queries))

    # 所有批次處理完畢，刪除進度檔案
    if (start_batch == 0 and num_batches == 1) or (start_batch + 1 >= num_batches):
        if progress_filename.exists():
            progress_filename.unlink()


def merge_data(json_dir: Path, output_csv_file: Path, fieldnames: List[str]) -> None:
    # 檢查JSON文件的目錄是否存在
    if not json_dir.exists():
        raise FileNotFoundError(f"The directory {json_dir} does not exist")

    # 搜集所有JSON文件的路徑
    json_files = [
        file_path for file_path in json_dir.iterdir() if file_path.suffix == ".json"
    ]

    # 將合併的數據寫入一個CSV文件，並使用tqdm顯示進度條
    with output_csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_path in tqdm(json_files, desc="Processing JSON files"):
            with file_path.open("r", encoding="utf-8") as file:
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

    base_path = Path("json_files_from_query_ner")
    similarity_search_path = base_path / "similarity_search_result"

    client = chromadb.HttpClient(
        host="127.0.0.1",
        port="8000",
    )

    collection = get_or_create_chroma_collection(
        collection_name="collect_cubelab_qa_lite"
    )

    # data必要欄位: ["category","變形問題"]
    data = pd.read_csv(base_path / "ner_results_by_LLM.csv")
    data.rename({"modify_query": "變形問題"}, axis=1, inplace=True)
    data = data.dropna(subset=["變形問題"])
    # 注意ner失敗的問題會補空值, modify_query 為空解析會出錯, 要移除
    process_batches(data, collection)
    merge_data(
        json_dir=similarity_search_path,
        output_csv_file=similarity_search_path / "merged_data_with_extra_q.csv",
        fieldnames=["問題類別", "變形問題", "相似度", "document"],
    )

    df = pd.read_csv(similarity_search_path / "merged_data_with_extra_q.csv")

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
                [
                    result,
                    (df_tmp[(df_tmp.相似度 == dmax) | (df_tmp.相似度 == dmin)]),
                ],
                axis=0,
            )
        result.reset_index(drop=True, inplace=True)

        return result

    # %%
    result_df = group_min_max(df, list("ABCDEFG"))

    # %%

    with pd.ExcelWriter(base_path / "output.xlsx", engine="xlsxwriter") as writer:
        df1.to_excel(writer, sheet_name="相似度", index=False)
        df2.to_excel(writer, sheet_name="by 問題類別", index=False)
        result_df.to_excel(writer, sheet_name="相似度最大最小值", index=False)

# %%