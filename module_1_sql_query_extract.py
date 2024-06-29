# %%

import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from utils_vectordb import get_or_create_http_chromadb
from utils_retriver import get_metadata_runnable, get_sql_querys


template = """
請根據'用戶問題'及'範例sql'進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
需要識別的命名實體是在'範例sql語法'中以&開頭的參數, 如&start_date, &end_date 等, 除了CustomerID所有參數都必須解析出來, 如無法判斷請返回None
解析出來的命名實體在第一碼加上&符號

例如解析出 start_date 就以 &start_date 作為key值

請注意用戶的問題中可能會包含以下類別的命名實體:時間如今天、明天、昨天...、商戶如uber, 蝦皮...、消費類別如食、衣、住、行...

一周的第一天是星期日

解析出來的日期請以8碼日期格式呈現, 例如"2024-05-01"請寫成20240501

時間請依照今日是{today}進行換算

用戶問題: {question}

標準問題: {標準問題}

範例sql:{SQL}
命名實體:
"""


def main():
    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )

    vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_00")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "lambda_mult": 0.25},
    )

    prompt = ChatPromptTemplate.from_template(template)

    json_parser_chain = prompt | model | JsonOutputParser()

    main_chain = {
        "標準問題": retriever | get_metadata_runnable("問題類別"),
        "SQL": retriever | get_metadata_runnable("SQL1", "SQL2", "SQL3"),
        "question": RunnablePassthrough(),
        "today": RunnableLambda(lambda x: "2024-05-01"),
    } | RunnablePassthrough.assign(keys=json_parser_chain)

    response = main_chain.invoke("過去一年我在優步、拉亞漢堡的消費數字?")

    querys = get_sql_querys(response=response, user_id="A")

    return querys, response


if __name__ == "__main__":
    q, r = main()


    # %%
    import pandas as pd
    import sqlite3

    query = """
    SELECT
        A.Consumption_Category_Desc AS `消費類別`,
        CAST(A.SUM_TXN_AMT AS FLOAT)  / CAST(B.TOT_TXN_AMT AS FLOAT) AS `消費佔比`
    FROM
        (
            SELECT
                CUSTOMERID,
                Consumption_Category_Desc,
                SUM(TXN_AMT) AS SUM_TXN_AMT
            FROM
                dual
            WHERE
                TXN_DATE BETWEEN 20230501 AND 20240501
                AND CUSTOMERID = 'A'
            GROUP BY
                CUSTOMERID,
                Consumption_Category_Desc
        ) A
        LEFT JOIN (
            SELECT
                CUSTOMERID,
                SUM(TXN_AMT) AS TOT_TXN_AMT
            FROM
                dual
            WHERE
                TXN_DATE BETWEEN 20230501 AND 20240501
                AND CUSTOMERID = 'A'
            GROUP BY
                CUSTOMERID
        ) B ON A.CUSTOMERID = B.CUSTOMERID
    """

    query = q[0]

    with sqlite3.connect("cubelab_txn.db") as conn:
        d = pd.read_sql(query, conn)


    d