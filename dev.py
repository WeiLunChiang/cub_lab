# %%
import os
from sqlite_create_and_insert import query_data_from_rdb

# %%
query = [
    'Select Count(*) From dual Where customer_id = "A" And TXN_DATE between "20240322" and "20240428" And MERCHANT_NAME_CLEAN IN( "統一超商")',
    'Select Sum(TXN_AMT) From dual Where customer_id = "A" And TXN_DATE between "20240322" and "20240428" And MERCHANT_NAME_CLEAN IN( "統一超商")',
    'Select TXN_DATE, TXN_AMT, MERCHANT_NAME_CLEAN From dual Where customer_id = "A" And TXN_DATE between "20240322" and "20240428" And MERCHANT_NAME_CLEAN IN( "統一超商")',
]
# %%
df_list = query_data_from_rdb(query, "cubelab_txn.db")
# query = """select * from dual where MERCHANT_NAME_CLEAN='統一超商'"""
# df = query_data_from_rdb(
#     """select * from dual where MERCHANT_NAME_CLEAN='統一超商'""", "cubelab_txn.db"
# )[0]
# df
# %% 資料拉回來 -> 回覆最終答案

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


import pandas as pd

pd.set_option("display.max_rows", None)  # 設定顯示無限多列
pd.set_option("display.max_columns", None)  # 設定顯示無限多行




# %%

template = """

請根據"用戶問題"把"response_1"、"response_2"、"response_3"包裝成一篇清晰且語氣輕鬆的"用戶回覆",
用戶回覆內容需要使用"我理解到您的問題是"加入原始的"用戶問題"。

用戶問題:\n 
{user_input} \n\n

LLM回覆:\n 

response_1:\n 
{response_1} \n\n

===========================================================
response_2:\n 
{response_2} \n\n

===========================================================
response_3:\n 
{response_3} \n\n
===========================================================


用戶回覆: \n
"""

# %%
prompt = ChatPromptTemplate.from_template(template)

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

parser = StrOutputParser()
# %%
chain = prompt | model | parser


response = chain.invoke(
    {
        "user_input": "請給我上個月在統一超商的消費紀錄",
        "response_1": df_list[0],
        "response_2": df_list[1],
        "response_3": df_list[2],
    }
)
# %%
print(response)
# %%
