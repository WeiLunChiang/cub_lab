import chromadb
import pandas as pd
import json
import time
from functools import wraps
from typing import Callable, Any, List, Dict
from utils_vectordb import get_or_create_chroma_collection
from pathlib import Path
from tqdm import tqdm

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper


# @timing_decorator
def query_db(
    queries: List[str],
    collection: Any = None,
    n_results: int = 20,
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
        query_texts=queries, include=["distances"], n_results=n_results
    )
    all_distances = results["distances"]
    all_cosine_similarities = [
        [abs(1 - distance) for distance in distances] for distances in all_distances
    ]
    return all_cosine_similarities


def save_results_to_json(filename: Path, results: List[Dict], progress: int):
    temp_filename = filename.with_suffix(".tmp")
    with temp_filename.open("w", encoding="utf-8") as f:
        json.dump(
            {"progress": progress, "results": results}, f, ensure_ascii=False, indent=4
        )
    temp_filename.replace(filename)


def load_results_from_json(filename: Path) -> Dict:
    if filename.exists():
        with filename.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"progress": 0, "results": []}


def process_batches(data: pd.DataFrame, collection: Any, batch_size: int = 1000):
    all_queries = list(data["變形問題"])
    all_categories = list(data["category"])
    results_file = Path("query_results.json")
    data = load_results_from_json(results_file)

    start_index = data["progress"]
    results = data["results"]

    num_batches = (len(all_queries) - start_index + batch_size - 1) // batch_size

    with tqdm(
        total=len(all_queries),
        initial=start_index,
        unit="queries",
        desc="Processing",
        ncols=100,
        leave=True,
    ) as pbar:
        for i in range(start_index, len(all_queries), batch_size):
            batch_queries = all_queries[i : i + batch_size]
            batch_categories = all_categories[i : i + batch_size]

            try:
                similarities = query_db(batch_queries, collection)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                continue

            for j in range(len(batch_queries)):
                result = {
                    "問題類別": batch_categories[j],
                    "用戶問題": batch_queries[j],
                    "相似度": similarities[j],
                }
                results.append(result)

            save_results_to_json(results_file, results, i + batch_size)
            pbar.update(batch_size)


if __name__ == "__main__":
    client = chromadb.HttpClient(
        host="127.0.0.1",
        port="8000",
    )

    data = pd.read_csv("./qa_set_with_sql.csv")

    collection = get_or_create_chroma_collection(
        collection_name="collect_cubelab_qa_00"
    )

    process_batches(data, collection)
