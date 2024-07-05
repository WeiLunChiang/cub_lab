import os
import sqlite3
import pandas as pd
from typing import Union, List


def query_data_from_rdb(
    querys: Union[List[str], str], db_name: str
) -> List[pd.DataFrame]:
    """
    Executes SQL queries against a SQLite database and returns the results as a list of DataFrames.

    Args:
    querys (Union[List[str], str]): SQL query or list of queries.
    db_name (str): Database file name.

    Returns:
    List[pd.DataFrame]: Results of the queries as DataFrames.

    Raises:
    ValueError: If the querys is empty.
    sqlite3.DatabaseError: For database connection or execution issues.

    Example:
    >>> querys = ["SELECT count(*) FROM dual", "SELECT * FROM dual"]
    >>> db_name = "example.db"
    >>> result = query_data_from_rdb(querys, db_name)
    >>> type(result[0])
    <class 'pandas.core.frame.DataFrame'>
    """
    if not os.path.exists(db_name):
        raise ValueError(f"The database {db_name} does not exist.")

    with sqlite3.connect(db_name) as conn:
        if isinstance(querys, str):
            d = [pd.read_sql_query(querys, conn)]
        elif len(querys) > 0:
            d = []
            for q in querys:
                d.append(pd.read_sql_query(q, conn))
        else:
            raise ValueError("The query list is empty.")

    return d


def main():
    import pandas as pd
    import sqlite3

    with sqlite3.connect("cubelab_txn.db") as conn:

        query = """DROP TABLE IF EXISTS DUAL"""

        conn.execute(query)
        conn.commit()

    with sqlite3.connect("cubelab_txn.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS dual (
            CustomerID TEXT,
            TXN_DATE TEXT,
            TXN_AMT INTEGER,
            MERCHANT_NAME TEXT,
            Consumption_Category_Desc TEXT
        )
        """
        )
        conn.commit()

    print("資料表 'dual' 已成功創建!")

    with sqlite3.connect("cubelab_txn.db") as conn:
        data = pd.read_excel("cubelab.xlsx", sheet_name="Raw_data")
        data["TXN_DATE"] = (
            data["TXN_DATE"].astype("string").apply(lambda x: x.replace("-", ""))
        )
        data.to_sql("dual", conn, if_exists="append", index=False)

    print(f"{data.shape[0]}筆資料資料已成功插入到 'dual' 表格!")

    # with sqlite3.connect("cubelab_txn.db") as conn:

    #     query = """
    #     select count(*) as `123`
    #     from dual
    #     where customer_id in('A','B')
    #     """
    #     df = pd.read_sql(query, conn)


if __name__ == "__main__":
    main()
