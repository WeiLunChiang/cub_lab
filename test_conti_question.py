# %%
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from chat_momery import runnable_with_history
from uuid import uuid4
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from module_1_sql_query_extract import template_2 as TEMPLATE
from dotenv import load_dotenv

load_dotenv(override=True)


# %%
class QuestionParser:
    def __init__(self, TEMPLATE):
        self._TEMPLATE = TEMPLATE
        self._init_chain1()
        self._init_chain2()

    @classmethod
    def create_session_id(cls):
        user_id = str(uuid4())
        return user_id

    def parse_question(self, user_input_raw, session_id):
        user_input = self.chain1.invoke(
            {"user_input": user_input_raw},
            config={"configurable": {"session_id": session_id}},
        )
        return user_input

    def ner_question(self, user_input):
        result = self.chain2.invoke(user_input)
        return result

    def _init_chain1(self):
        self.chain1 = runnable_with_history

    def _init_chain2(self):
        prompt = ChatPromptTemplate.from_template(self._TEMPLATE)
        model = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            temperature=0.0,
        )
        chain = prompt | model | JsonOutputParser()
        self.chain2 = chain

    def run(self, rid, q1, q2):
        result = {}
        session_id = self.create_session_id()
        result["session_id"] = session_id
        result["rid"] = rid
        result["question_q1"] = {"q_ori": q1}
        result["question_q2"] = {"q_ori": q2}

        user_input_1 = self.parse_question(q1, session_id)
        result["question_q1"]["q_parse"] = user_input_1
        result["result_q1"] = self.ner_question(user_input_1)

        user_input_2 = self.parse_question(q2, session_id)
        result["question_q2"]["q_parse"] = user_input_2
        result["result_q2"] = self.ner_question(user_input_2)
        return result


# %%
def save_progress(temp_dir, batch_index, batch_result):
    """保存每個批次的結果到獨立的臨時檔案."""
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_progress_batch_{batch_index}.json")
    with open(temp_path, "w") as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=4)


def load_progress(temp_dir):
    """讀取所有已保存的批次結果，並返回已處理的批次數量."""
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        print(f"mkdir {temp_dir}")

    processed_batches = sorted(
        [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(temp_dir)
            if f.startswith("temp_progress_batch_")
        ]
    )
    return processed_batches


def load_batch_result(temp_dir, batch_index):
    """讀取指定批次的結果."""
    temp_path = os.path.join(temp_dir, f"temp_progress_batch_{batch_index}.json")
    with open(temp_path, "r") as f:
        return json.load(f)


def merge_results(temp_dir, num_batches):
    """將所有批次結果合併為一個大結果."""
    result = []
    for i in range(num_batches):
        batch_result = load_batch_result(temp_dir, i)
        result.extend(batch_result)
    return result


def log_failed_batch(failed_batches, rid, q1, q2, error_message):
    """記錄失敗的批次及錯誤信息."""
    failed_batches.append({"rid": rid, "q1": q1, "q2": q2, "error": str(error_message)})


def save_failed_batches(failed_batches, temp_dir):
    """保存失敗的批次到單獨的JSON檔案."""
    failed_path = os.path.join(temp_dir, "failed_batches.json")
    with open(failed_path, "w") as f:
        json.dump(failed_batches, f, ensure_ascii=False, indent=4)


def main():
    DATA_PATH = "test/test_data_continue_sample.csv"
    OUTPUT_PATH = "test/conti_question.json"
    TEMP_DIR = "test/temp_progress"  # 用於存儲暫存結果的目錄
    BATCH_SIZE = 10

    # 載入資料
    test_data = pd.read_csv(DATA_PATH)
    parser = QuestionParser(TEMPLATE)

    # 將資料分成每10筆為一個batch
    batches = np.array_split(
        test_data[["rid", "q1", "q2"]].values, len(test_data) // BATCH_SIZE + 1
    )

    # 讀取之前的進度
    processed_batches = load_progress(TEMP_DIR)
    start_batch = processed_batches[-1] + 1 if processed_batches else 0
    failed_batches = []  # 用於記錄失敗的批次

    # 顯示進度條
    for i, batch in tqdm(
        enumerate(batches[start_batch:], start=start_batch),
        total=len(batches),
        desc="Processing batches",
    ):
        batch_result = []  # 改為列表，將每筆結果分開存儲
        for rid, q1, q2 in batch:
            try:
                # 嘗試處理批次
                result = parser.run(rid=rid, q1=q1, q2=q2)
                batch_result.append(result)  # 將每個結果加入列表中
            except Exception as e:
                # 記錄失敗批次
                log_failed_batch(failed_batches, rid, q1, q2, e)
                continue  # 繼續下一筆資料

        # 保存每個批次的結果到獨立檔案
        save_progress(TEMP_DIR, i, batch_result)

    # 保存失敗的批次
    if failed_batches:
        save_failed_batches(failed_batches, TEMP_DIR)

    # 合併所有批次的結果
    result = merge_results(TEMP_DIR, len(batches))

    # 最後保存完整結果
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # 清除臨時檔案資料夾及檔案
    for file in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file))
    os.rmdir(TEMP_DIR)


if __name__ == "__main__":
    main()
