# %%
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from module_1_sql_query_extract import template_1, template_2

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "300"

# %% 用戶發問-> 打LLM ner -> 儲存成json -> 整理成csv

df = pd.read_csv("qa_set_with_sql.csv")
df_A_C_D = df[df["category"].isin(["A", "D"])]

# 隨機抽取 category 為 B 的 20000 筆資料
df_B_sample = df[df["category"] == "B"].sample(n=20000, random_state=1, replace=False)

# 合併所有篩選和抽樣的資料
df_sample = pd.concat([df_A_C_D, df_B_sample], ignore_index=True)

# 手動添加的問句
financial_questions = [
    "我可以申請房屋貸款嗎？",
    "定期存款的利率是多少？",
    "如何開設投資帳戶？",
    "請問有沒有推薦的保險產品？",
    "如何進行基金定投？",
    "股票交易的手續費是多少？",
    "如何申請小額信貸？",
    "我能申請外幣存款帳戶嗎？",
    "買黃金有什麼投資策略？",
    "請問如何計算貸款的月供？",
    "教育儲蓄保險有哪些選擇？",
    "如何進行理財規劃？",
    "退休金計劃有什麼選擇？",
    "債券投資的風險有哪些？",
    "如何申請汽車貸款？",
    "健康保險的保障範圍有哪些？",
    "請問如何開始投資股票？",
    "外匯交易的基本知識是什麼？",
    "如何計算我的財務狀況？",
    "金融理財產品的風險等級如何區分？",
]

life_questions = [
    "你今天過得怎麼樣？",
    "你最喜歡的興趣愛好是什麼？",
    "平時都喜歡做哪些運動？",
    "你有養寵物嗎？",
    "你最近在看什麼書？",
    "你對旅遊有什麼計劃？",
    "你喜歡做飯嗎？",
    "你最喜歡的電影是哪一部？",
    "你的生日是哪天？",
    "你住在哪個城市？",
    "你有兄弟姐妹嗎？",
    "你通常怎麼放鬆自己？",
    "你最喜歡的音樂類型是什麼？",
    "你最近在追哪些電視劇？",
    "你有什麼特別的技能嗎？",
    "你最喜歡的節日是什麼？",
    "你喜歡喝咖啡還是茶？",
    "你通常怎麼過週末？",
    "你對健康飲食有什麼看法？",
    "你有學過什麼樂器嗎？",
]

credit_card_questions = [
    "上個月我在蝦皮花了多少錢？",
    "我上次在全聯刷了多少錢？",
    "今年到目前為止我在加油站消費了多少？",
    "我上次在大潤發的消費金額是多少？",
    "最近一次在星巴克的消費是多少？",
    "我的信用卡這個月在餐廳消費了多少錢？",
    "上週我在家樂福的消費金額是多少？",
    "我在去年12月的總消費金額是多少？",
    "昨天我在誠品書店刷了多少錢？",
    "最近一次在便利商店的消費是多少？",
    "我上個月的娛樂支出總額是多少？",
    "我在上次電影院的消費是多少？",
    "最近一次在藥局的消費是多少？",
    "上個月我在網購上的總消費是多少？",
    "我最近一次在餐廳的消費是多少？",
    "我的信用卡上次繳了多少水電費？",
    "我上週的總消費金額是多少？",
    "最近一次在百貨公司的消費是多少？",
    "今年我在服飾店的總消費是多少？",
    "上個月我在外送平台花了多少錢？",
]

# 創建手動添加的問句的 DataFrame
additional_df = pd.DataFrame(
    {
        "category": ["E"] * len(financial_questions)
        + ["F"] * len(life_questions)
        + ["G"] * len(credit_card_questions),
        "變形問題": financial_questions + life_questions + credit_card_questions,
    }
)

# 將手動添加的問句合併到原始資料集中
df_sample = pd.concat([df_sample, additional_df], ignore_index=True)
user_input_list = df_sample["變形問題"].tolist()
category_list = df_sample["category"].tolist()

# %% 2. 比對VDB 拿到SQL query
model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

prompt = ChatPromptTemplate.from_template(template_2)

json_parser_chain = prompt | model | JsonOutputParser()

chain_1 = {
    "question": RunnablePassthrough(),
    "today": RunnableLambda(lambda x: "2024-05-01"),
} | json_parser_chain


def save_results_to_json(filename: Path, results: list):
    with filename.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def load_results_from_json(filename: Path):
    if filename.exists():
        with filename.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"progress": 0, "results": []}


def process_batches(user_inputs: list, categories: list, batch_size: int = 200):
    results_dir = Path("json_files_from_query_ner")
    results_dir.mkdir(exist_ok=True)

    progress_file = results_dir / "progress.json"
    progress_data = load_results_from_json(progress_file)

    start_index = progress_data["progress"]
    all_results = progress_data["results"]

    with tqdm(
        total=len(user_inputs),
        initial=start_index,
        unit="queries",
        desc="Processing",
        ncols=100,
        leave=True,
    ) as pbar:
        for i in range(start_index, len(user_inputs), batch_size):
            batch_inputs = user_inputs[i : i + batch_size]
            batch_categories = categories[i : i + batch_size]

            try:
                response = chain_1.batch(batch_inputs)
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}")
                continue

            batch_results = []
            for query, category, resp in zip(batch_inputs, batch_categories, response):
                batch_results.append(
                    {"category": category, "query": query, "response": resp}
                )

            batch_filename = results_dir / f"batch_{i // batch_size + 1}.json"
            save_results_to_json(batch_filename, batch_results)

            progress_data = {"progress": i + batch_size, "results": all_results}
            save_results_to_json(progress_file, progress_data)

            pbar.update(len(batch_inputs))


def combine_results_from_json(results_dir: Path, output_csv: Path):
    all_results = []
    for json_file in results_dir.glob("batch_*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            batch_results = json.load(f)
            all_results.extend(batch_results)

    df = pd.DataFrame(all_results)
    df["modify_query"] = [
        ix["response"].get("modify_query", None) for ix in all_results
    ]
    df = df[["category", "query", "modify_query", "response"]]
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_batches(user_input_list, category_list)
    combine_results_from_json(
        Path("json_files_from_query_ner"), Path("json_files_from_query_ner/ner_results_by_LLM.csv")
    )

    # %%
    import pandas as pd

    df = pd.read_csv("json_files_from_query_ner/ner_results_by_LLM.csv")
