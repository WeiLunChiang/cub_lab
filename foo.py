# %%
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils_vectordb import (
    get_or_create_http_chromadb,
    get_or_create_chroma_http_collection,
)

# %% template
template = """
請根據用戶問題進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
    請參照以下範例: 
            問題:
            我過去一個月在gogoro吃飯的金額是多少?
        回答:
            變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
            標準問句: [時間]在[商戶]的[消費類型]的消費紀錄'\    \       \   kjlkkjkkkmkljkmmmmlk;lkl,m.,m,.,mjjklkkkjnhbm32w32w32w32w323rterlt[p]
            命名實體: 'start_date': '2024-04-25', 'end_date': '2024-05-25', '商戶': 'gogoro', '類別': '食'

        
    請注意用戶的問題中可能會包含以下類別的命名實體:時間(如今天、明天、昨天...)、商戶(如uber, 蝦皮...)、類別(如食、衣、住、行...)
    
    時間請依照今日是{today}進行換算

    上個月就是過去一個月，必須包含今日，前月就是過去一個月，必須包含今日, 上週就是過去七天(包含今日)

    各類別、全部類別、...都是一種[類別]

    問題: {question}
    回答:
        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄

        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄

        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄
        命名實體:
"""


# %%
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
from utils_retriver import get_metadata_runnable


# %%
vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_00")
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.25},
)
# %%


template_1 = """
請根據'用戶問題'及'範例sql'進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
需要識別的命名實體是在'範例sql語法'中以&開頭的參數, 如&start_date, &end_date 等, 除了CustomerID所有參數都必須解析出來, 如無法判斷請返回None
解析出來的命名實體在第一碼加上&符號
裡如解析出 start_date 就以 &start_date 作為key值

請注意用戶的問題中可能會包含以下類別的命名實體:時間如今天、明天、昨天...、商戶如uber, 蝦皮...、消費類別如食、衣、住、行...

時間請依照今日是{today}進行換算

用戶問題: {question}

標準問題: {標準問題}

範例sql:{SQL}

命名實體:
"""

json_parser_chain = ChatPromptTemplate.from_template(template_1) | model | JsonOutputParser() 

main_chain = (
    RunnableParallel(
        {
            "標準問題": retriever
            | get_metadata_runnable(
                "問題類別"
            ),  # rag metadata 問題類別 <- 要給LLM當作參考,
            "SQL": retriever
            | get_metadata_runnable(
                "SQL1", "SQL2", "SQL3"
            ),  # rag metadata SQL <- 要給LLM當作參考,
            "question": RunnablePassthrough(),  # 原始問題
            "today": RunnableLambda(lambda x: "2024-05-01"),
        }
    )
    | RunnablePassthrough.assign(output = json_parser_chain)
)

response = main_chain.invoke("上周在sogo百貨的刷卡金額")
response


# %%




# %%
rp = RunnableParallel(
        {
            "標準問題": retriever
            | get_metadata_runnable(
                "問題類別"
            ),  # rag metadata 問題類別 <- 要給LLM當作參考,
            "SQL": retriever
            | get_metadata_runnable(
                "SQL1", "SQL2", "SQL3"
            ),  # rag metadata SQL <- 要給LLM當作參考,
            "question": RunnablePassthrough(),  # 原始問題 <- 要請LLM解析
            "today": RunnableLambda(lambda x: "2024-05-01"),
        }
    )
# %%
chain_0 = RunnableParallel(context = retriever, question = RunnablePassthrough() )
# |   {
#             "標準問題": get_metadata_runnable("問題類別"),  # rag metadata 問題類別 <- 要給LLM當作參考,
#             "SQL": get_metadata_runnable("SQL1", "SQL2", "SQL3"),  # rag metadata SQL <- 要給LLM當作參考,
#         }
chain_0.invoke('上周在sogo百貨的刷卡金額')
# %%

ChatPromptTemplate.from_template('{a}').invoke({'a':123})
# %%
chain_1 = RunnableParallel(
        {
            "標準問題": get_metadata_runnable("問題類別"),  # rag metadata 問題類別 <- 要給LLM當作參考,
            "SQL": get_metadata_runnable("SQL1", "SQL2", "SQL3"),  # rag metadata SQL <- 要給LLM當作參考,
        }
    )

response_1 = chain_1.invoke(response_0)

# %%

chain_2 = 

            "question": RunnablePassthrough(),  # 原始問題 <- 要請LLM解析
            "today": RunnableLambda(lambda x: "2024-05-01"),
# %%
response_1 = rp.invoke("上周在sogo百貨的刷卡金額")



chain_2 = ChatPromptTemplate.from_template(template_1) | model | JsonOutputParser()

chain_2.invoke(response_1)


# %%
"""
解析出SQL參數 -> 填入標準SQL
"""

text = response
d_keys = response.keys()

for k in d_keys:


# %%


chain = retriever | get_metadata_runnable("SQL1", "SQL2", "SQL3")
response_sql = chain.invoke("上周在sogo百貨的刷卡金額")

# %%

response

replace_keys = [ix for ix in response_sql[0].split(' ') if ix.startswith('&')]
replace_keys
# %%
response_sql[0].replace(replace_keys[1], response.get(replace_keys[1]))

# %%

template = """
請根據用戶問題進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
    請參照以下範例: 
            問題:
            我過去一個月在gogoro吃飯的金額是多少?
        回答:
            變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
            標準問句: [時間]在[商戶]的[消費類型]的消費紀錄
            命名實體: 'start_date': '2024-04-25', 'end_date': '2024-05-25', '商戶': 'gogoro', '類別': '食'

        
    請注意用戶的問題中可能會包含以下類別的命名實體:時間(如今天、明天、昨天...)、商戶(如uber, 蝦皮...)、類別(如食、衣、住、行...)
    
    時間請依照今日是'2024-05-01'進行換算

    上個月就是過去一個月，必須包含今日，前月就是過去一個月，必須包含今日, 上週就是過去七天(包含今日)

    各類別、全部類別、...都是一種[類別]

    問題: {question}
    回答:
        {}
        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄

        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄

        變形問句: 我[時間]在[商戶][消費類型]的金額是多少?
        標準問句: [時間]在[商戶]的[消費類型]的消費紀錄
        命名實體:
"""

# %%
