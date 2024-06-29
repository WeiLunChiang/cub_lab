# %%
import sys
import os
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from utils_vectordb import get_or_create_http_chromadb
from utils_retriver import get_metadata_runnable, get_sql_querys
from langchain_core.output_parsers import StrOutputParser


from module_1_sql_query_extract import template as template_1
from sqlite_create_and_insert import query_data_from_rdb

from chat_momery import runnable_with_history


"""
簡要流程:
1. 用戶發問
1.5 潤飾成完整問句
2. 比對VDB 拿到SQL
3. SQL拉RDB 拿到資料
4. 資料+回答潤飾 拿到回答
"""

# %% 1.用戶發問
today = "2023-05-08"
user_id = "A"

# user_input = sys.argv[1]
session_id = "996"
user_input_raw = "那sogo百貨呢"  # 測試一至多家商家的取代狀況

# %% 1.5 潤飾成完整問句

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

vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_00")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.25},
)

prompt = ChatPromptTemplate.from_template(template_1)

json_parser_chain = prompt | model | JsonOutputParser()

chain = {
    "標準問題": retriever | get_metadata_runnable("問題類別"),
    "SQL": retriever | get_metadata_runnable("SQL1", "SQL2", "SQL3"),
    "question": RunnablePassthrough(),
    "today": RunnableLambda(lambda x: today),
} | RunnablePassthrough.assign(keys=json_parser_chain)

response = chain.invoke(user_input)
# breakpoint()
try:
    querys = get_sql_querys(response=response, user_id=user_id)
except:
    breakpoint()
# %% SQL拉RDB 拿到資料
df_list = query_data_from_rdb(querys, "cubelab_txn.db")

# %% 資料+回答潤飾 拿到回答

from module2_llm_response import template as template_2


# %%
prompt = ChatPromptTemplate.from_template(template_2)
parser = StrOutputParser()
# %%
chain = prompt | model | parser


response = chain.invoke(
    {
        "user_input": user_input,
        "response_1": df_list[0],
        "response_2": df_list[1],
        "response_3": df_list[2],
    }
)
# %%
print(response)

# %%