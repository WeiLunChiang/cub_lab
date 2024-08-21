# %% chrmoa 啟動+安裝語法
import os
import mlflow
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from utils_retriever import get_metadata_runnable, get_sql_querys, RetrieveWithScore
from module_1_sql_query_extract import template_1, template_2
from create_and_insert_sqlite import query_data_from_rdb
from chat_momery import runnable_with_history
from icecream import ic as print
from module2_llm_response import template as template_3
from module2_llm_response import cust_desc_dict
from utils_vectordb import get_or_create_http_chromadb
from utils_logging import mlflow_exception_logger, mlflow_openai_callback


@mlflow_exception_logger
@mlflow_openai_callback
def main():

    # %%
    """
    簡要流程:
    1. 用戶發問
    1.5 潤飾成完整問句
    2. 比對VDB 拿到SQL
    3. SQL拉RDB 拿到資料
    4. 資料+回答潤飾 拿到回答
    """

    # %% 1.用戶發問
    today = "2023-06-08"
    user_id = "A"
    session_id = "0099"
    user_input_raw = "過去一年我在蝦皮、優食花了多少錢?"  # "請給我過去一年的消費紀錄" "我過去一年花了多少錢"<- 這句沒過

    # %% 1.5 完整問句

    user_input = runnable_with_history.invoke(
        {"user_input": user_input_raw},
        config={"configurable": {"session_id": session_id}},
    )
    print(user_input)

    # %% 2. 比對VDB 拿到SQL query
    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )

    vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_lite")

    retriever = RetrieveWithScore(vectorstore, k=3, score_threshold=0)
    # retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(template_2)

    json_parser_chain = prompt | model | JsonOutputParser()

    retriever_chain = (
        RunnableLambda(lambda response: response["modify_query"])
        | retriever
        | {  # vectordb拉到的內容(包含SQL)
            "SQL": get_metadata_runnable("SQL1", "SQL2", "SQL3"),
            "標準問題": get_metadata_runnable("問題類別"),
            "category": get_metadata_runnable("category"),
            "score": get_metadata_runnable("score"),
        }
    )

    chain_1 = (
        {
            "question": RunnablePassthrough(),
            "today": RunnableLambda(lambda x: "2024-05-01"),
        }
        | json_parser_chain
        | {
            "keys": RunnablePassthrough(),  # 解析出來的參數
            "retriever": retriever_chain,
        }
    )

    response = chain_1.invoke(user_input)

    querys = get_sql_querys(response=response, user_id=user_id)
    print(querys)

    # %% SQL拉RDB 拿到資料

    # get data from sqlite db
    df_list = query_data_from_rdb(querys, "cubelab_txn.db")

    # %% 資料+回答潤飾 拿到回答

    prompt = ChatPromptTemplate.from_template(template_3)
    parser = StrOutputParser()
    # %%
    chain = prompt | model | parser

    tones = ["高端VIP用戶", "一般用戶", "態度不佳用戶"]
    # 打LLM
    for tone in tones:
        print(tone)
        response = chain.invoke(
            {
                "user_input": user_input,
                "response_1": df_list[0],
                "response_2": df_list[1],
                # "response_3": df_list[2],
                "tone": tone,
                "desc": cust_desc_dict.get(tone),
            }
        )
        # %%
        print(response)

    # %%


if __name__ == "__main__":
    os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "30"

    mlflow.set_experiment("cubelab_demo")
    mlflow.langchain.autolog()
    with mlflow.start_run() as run:
        main()
