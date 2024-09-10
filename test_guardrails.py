# %%
import pandas as pd

dt = pd.read_excel("cubelab.xlsx", sheet_name="時間")
# %%
d_merge = d_ori[
    ["編號", "問題類別", "變形問題", "time", "store_name", "category"]
].merge(
    dt[["替代變數", "start_date", "end_date"]],
    left_on="time",
    right_on="替代變數",
    how="left",
)
# %%
d_merge[
    [
        "編號",
        "問題類別",
        "變形問題",
        "time",
        "start_date",
        "end_date",
        "store_name",
        "category",
    ]
].to_csv("test/test_data2.csv", index=False)
# %%
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template(template_2)

model_with_guardrails = AzureChatOpenAI(
    api_key="Bearer example_token",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint="http://a92628a37f448457aade14d9be6d7d36-1492598554.ap-northeast-1.elb.amazonaws.com",
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    temperature=0.0,
)
prompt = ChatPromptTemplate.from_template("{input}")
chain = prompt | model_with_guardrails
queries = [
    "那次在2023年的購物活動，我總共投入了多少金錢?",
    "說明2023/5/15我在加油站的開支了多少??",
    "我在本週光顧悠遊加值的時候，總共消費了多少錢財?",
    "2022/11期間，我在全國電子那裡購物的總開支金額是多少?",
    "去年10月我在安聯人壽保險的消費概況?",
    "描述一下去年1/1-1/15我在YVES SAINT LAURENT MAC消費的數目。",
    "請告訴我，在今年第四季度我在電子化繳費的開銷是多少?",
    "那次在2022/11期間我在原來是洋蔥購物，總計支付了多少費用?",
    "描述一下今年我在好食集共營平台消費的數目。",
    "2023年我在鮨天本登龍門開銷的概況。",
]


r = []
for query in queries:
    r.append(chain.invoke({"input": query}))
    print(r[-1])

# %%

