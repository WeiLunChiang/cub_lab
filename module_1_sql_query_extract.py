# %%

import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from utils_vectordb import get_or_create_http_chromadb
from utils_retriever import get_metadata_runnable, get_sql_querys
from icecream import ic as print

template_1 = """
請根據'用戶問題'及'範例sql'進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
需要識別的命名實體是在'範例sql語法'中以&開頭的參數, 如&start_date, &end_date 等, 除了CustomerID所有參數都必須解析出來, 如無法判斷請返回None
解析出來的命名實體在第一碼加上&符號

例如解析出 start_date 就以 &start_date 作為key值

請注意用戶的問題中可能會包含以下類別的命名實體:時間如今天、明天、昨天...、商戶如uber, 蝦皮...、消費類別如餐飲、百貨、休閒等...
解析出來的消費類別不要包含"類"字, 例如"百貨類"請以"百貨"表示
解析出來的商家名稱要與輸入用字完全一致, 不要發生中英文改寫、縮寫等修改的狀況

一周的第一天是星期日

解析出來的日期請以8碼日期格式呈現, 例如"2024-05-01"請寫成20240501

時間請依照今日是{today}進行換算

用戶問題: {question}

標準問題: {標準問題}

範例sql:{SQL}
命名實體:
"""

template_2 = """
請對用戶問題進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
請以json格式返回識別到的內容, key值請遵照下列原則給定:
    時間: "&start_date","&end_date"
    商家: "&string1","&string2","&string3",...(依照用戶提到的商家名稱可能有0~n個)
    消費類別: "&string" 
    用戶問題挖空命名實體後的結果: "modify_query", 例如'請給我上個月在蝦皮的消費紀錄'辨識出'時間'、'商家',就要改寫成 '請給我[時間]在[商家]的消費紀錄'
綜合上述, 合法key值的選項只有["modify_query",""&start_date","&end_date","&string","&string1","&string2","&string3",...,"&string999"]

時間需依照今天日期轉換為開始時間、結束時間
一周的第一天是星期日

時間請一定要挖空成[時間],請注意時間副詞如過去、現在、近, 都是[時間]的一部分
例如'過去[時間]我在[商家]的消費數字?'就不對, 應該是'[時間]我在[商家]的消費數字?'

請注意用戶的問題中可能會包含以下類別的命名實體:時間如今天、明天、昨天...、商戶如uber, 蝦皮...、消費類別如餐飲、百貨、休閒等...
用戶對消費類別的描述可能會不精確, 如"餐飲"可能會以"飲食"、"吃飯"等形式出現, 其他消費類別也同理
消費類別前容易出現"買"、"在"等字眼如"買衣服"就是指"在服飾/鞋/精品類"、"買保險"就是指"在保險類"、"在餐飲類", 
消費類別後方容易以"類"連接如"餐飲類"就是"餐飲"，"休閒類"就是"休閒"

解析出來的消費類別不要包含"類"字, 例如"百貨類"請以"百貨"表示
解析出來的商家名稱要與輸入用字完全一致, 不要發生中英文改寫、縮寫等修改的狀況


解析出來的日期請以8碼日期格式呈現, 例如"2024-05-01"請寫成20240501


可能的"消費類別"清單如下:
飯店/住宿
餐飲
服飾/鞋/精品
休閒
旅遊
交通/運輸
悠遊卡加值
加油站
3C通訊家電
百貨
超市/量販
一般購物(食品)
一般購物(家具家飾裝潢/雜貨)
一般購物(其他)
分期付款
美容/美髮/保養
預借現金/貸款
捐獻
醫療救護/整型
公共事業費用
保險
繳稅
教育/學費
郵購/直銷
藝文
其他
其他類_NOTFOUND




時間請依照今日是{today}進行換算

用戶問題: {question}

命名實體:
"""




def main():
    model = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )

    # vectorstore = get_or_create_http_chromadb(collection_name="collect_cubelab_qa_lite_2")
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 1, "metric": "cosine"}
    # )

    prompt = ChatPromptTemplate.from_template(template_2)

    json_parser_chain = prompt | model | JsonOutputParser()

    main_chain = {
        "question": RunnablePassthrough(),
        "today": RunnableLambda(lambda x: "2024-05-01"),
    } | RunnablePassthrough.assign(keys=json_parser_chain)

    response = main_chain.invoke("去年在蝦皮跟sogo百貨花多少錢?")
    return response
    # querys = get_sql_querys(response=response, user_id="A")

    return querys, response


if __name__ == "__main__":
    res = main()
    print(res)