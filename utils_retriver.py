# %%
from collections import Counter
from langchain_core.runnables import RunnableLambda
from typing import List, Callable, Dict
from operator import itemgetter
import re


# %%
def _get_metadata(response: List, *keys: str) -> List:
    """
    Retrieves metadata based on provided keys from the response.

    If the response contains only one element, it directly retrieves
    the value corresponding to the keys from the metadata of that element.
    If the response contains multiple elements, it retrieves the value
    corresponding to the keys from the metadata of each element, and returns
    the value that appears the most times.

    Parameters:
    response -- A list containing metadata
    *keys -- The keys to retrieve from the metadata

    Returns:
    A list containing the values retrieved from the metadata.
    """
    if len(response) == 1:
        res = itemgetter(*keys)(response[0].metadata)
    else:
        lst = [itemgetter(*keys)(ix.metadata) for ix in response]
        counter = Counter(lst)
        res = counter.most_common(1)[0][0]

    if len(keys) == 1:
        return [res]
    else:
        return list(res)


def _create_partial_func(func: Callable, *keys: str) -> Callable:
    """
    Creates a partial function that calls the original function
    with the provided keys when invoked.

    Parameters:
    func -- The original function
    *keys -- The keys to pass to the original function

    Returns:
    A partial function that calls the original function
    with the provided keys when invoked.
    """

    def partial_func(response):
        return func(response, *keys)

    return partial_func


def get_metadata_runnable(*keys: str) -> Callable[[List], List]:
    """
    Creates a langchain runnable with the provided keys when invoked.

    Parameters:
    *keys -- The keys to pass to the function

    Returns:
    A langchain runnable function with the provided keys when invoked.
    """
    f = _create_partial_func(_get_metadata, *keys)
    return RunnableLambda(f)


def replace_sql_query(query: str, keys: Dict[str, str]) -> str:
    """
    Replace placeholders and keys in the SQL query.

    Args:
        query (str): The SQL query string.
        keys (Dict[str, str]): Dictionary with keys and values for replacement.

    Returns:
        str: Modified SQL query string.
    """
    # breakpoint()
    merch_keys = [ix for ix in keys.keys() if ("string" in ix)]

    if merch_keys:
        # 取商家名稱
        merch_names = ", ".join(['"' + keys[key] + '"' for key in merch_keys])
        query = query.replace("&string1, &string2, ...", "{merch_names}")
        query = query.format(merch_names=merch_names)

    for key in keys:
        if key not in merch_keys and key in query:
            # 商家名稱已經處理過, 應跳過不重複處理
            query = query.replace(key, '"' + keys.get(key) + '"')

    # 處理query中剩下沒換到的key
    keys_to_be_remove = [ix for ix in query.split(" ") if ix.startswith("&")]
    for key in keys_to_be_remove:
        if key in query:
            query = query.replace(key, "")

    # 字串取代 ", ...)" 取代為 ")"
    # query = re.sub(r",\s*\.\.\.\)", ")", query)

    # 處理in 中只有單一值的問題
    # breakpoint()
    pattern = re.compile(r'\bIN\s*\(\s*"([^"]+)"\s*\)')
    query = pattern.sub(lambda m: f'= "{m.group(1)}"', query)
    return query


# 測試一家、多家商家的取代狀況 確保不會出現IN("FOO")的不合法SQL query


def get_sql_querys(response: Dict[str, Dict[str, str]], user_id: str) -> List[str]:
    """
    Get processed SQL queries based on the response and customer ID.

    Args:
        response (Dict[str, Dict[str, str]]):
            A dictionary containing keys, values, and SQL queries.
        user_id (str):
            The customer ID to replace placeholders in the queries.

    Returns:
        List[str]: List of processed SQL queries.
    """

    keys = response["keys"]
    keys["&CustomerID"] = user_id

    querys = []
    for query in response["SQL"]:
        querys.append(replace_sql_query(query, keys))

    return querys
