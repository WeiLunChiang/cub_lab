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


template_0 = """

請根據"用戶問題"把"response_1"、"response_2"、"response_3"包裝成一篇清晰的"用戶回覆",
用戶回覆開頭需要使用"我理解到您的問題是"加入原始的"用戶問題"，且需使用{tone}的語氣進行一段開場(至少三句話)。
末尾需要加註警語"本回答是由AI助理生成, 可能存在一定機率不準確, 請自行確認"

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


template = """

你是有20年以上的資深產品經理，請以信用卡消費貢獻度與用戶語調的情緒將用戶拆分成三大客群(高端VIP/一般用戶/態度不佳用戶)。

目前面對一位屬於 {tone} 客群的客戶,
客群描述如下: 

{desc}

===========================================================
用戶問題:\n 
{user_input} \n\n

LLM回覆:\n 

response_1:\n 
{response_1} \n\n

===========================================================
response_2:\n 
{response_2} \n\n

===========================================================


請根據"用戶問題"把"response_1"、"response_2"包裝成一篇清晰的"用戶回覆",
用戶回覆開頭需要使用"我理解到您的問題是"加入原始的"用戶問題"，且需依照客戶所屬客群，使用最適當的回覆的語氣進行客製化回覆(客製化部分盡量簡潔)。
請注意使用繁體中文回覆，也不要出現中國式用語。
末尾需要加註警語"本回答是由AI助理生成, 可能存在一定機率不準確, 請自行確認"


用戶回覆: \n
"""

cust_desc_dict = {
    "高端VIP用戶": """特性:
    - 高消費貢獻度
    - 對專屬服務有較高需求
    - 常使用高額信用卡消費
    語調情緒:
    - 穩重、理性
    - 期待專業與尊重的語調
    回應策略:
    - 提供個性化的消費建議
    - 回應專業且有禮貌，解釋可參考並使用專屬優惠與獎勵""",
    "一般用戶": """    特性:
    - 中等消費貢獻度
    - 偶爾使用信用卡消費
    - 對價格敏感
    語調情緒:
    - 親切、友好
    - 期待簡單易懂的回應
    回應策略:
    - 提供日常消費的建議與優惠
    - 回應輕鬆且親切，解釋可參考並使用信用卡福利""",
    "態度不佳用戶": """特性:
    - 低消費貢獻度
    - 對信用卡服務有抱怨或不滿
    - 少使用信用卡消費
    語調情緒:
    - 負面、抱怨
    - 期待快速解決問題
    回應策略:
    - 主動查詢並解決用戶問題
    - 回應中立且尊重，強調解決方案""",
}


if __name__ == "__main__":

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
