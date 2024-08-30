# %% chrmoa 啟動+安裝語法
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from icecream import ic as print


template_2 = """
請對用戶問題進行命名實體識別(ner), 並嚴格以json格式只返回'命名實體'。
請以json格式返回識別到的內容, key值請遵照下列原則給定:
    時間: "&start_date","&end_date"
    商戶: "&string1","&string2","&string3",...(依照用戶提到的商戶名稱可能有0~n個)
    類別: "&string" 
    用戶問題挖空命名實體後的結果: "modify_query", 例如'請給我上個月在蝦皮的消費紀錄'辨識出'時間'、'商戶',就要改寫成 '請給我[時間]在[商戶]的消費紀錄'
綜合上述, 合法key值的選項只有["modify_query",""&start_date","&end_date","&string","&string1","&string2","&string3",...,"&string999"]

時間需依照今天日期轉換為開始時間、結束時間
一周的第一天是星期日

時間請一定要挖空成[時間],請注意時間副詞如過去、現在、近, 都是[時間]的一部分
例如'過去[時間]我在[商戶]的消費數字?'就不對, 應該是'[時間]我在[商戶]的消費數字?'

請注意用戶的問題中可能會包含以下類別的命名實體:時間如今天、明天、昨天...、商戶如uber, 蝦皮...、消費類別如餐飲、百貨、休閒等...
用戶對消費類別的描述可能會不精確, 如"餐飲"可能會以"飲食"、"吃飯"等形式出現, 其他消費類別也同理
消費類別前容易出現"買"、"在"等字眼如"買衣服"就是指"在服飾/鞋/精品類"、"買保險"就是指"在保險類"、"在餐飲類", 
消費類別後方容易以"類"連接如"餐飲類"就是"餐飲"，"休閒類"就是"休閒"

解析出來的消費類別不要包含"類"字, 例如"百貨類"請以"百貨"表示
解析出來的商戶名稱要與輸入用字完全一致, 不要發生中英文改寫、縮寫等修改的狀況


解析出來的日期請以YYYY/MM/DD日期格式呈現, 例如"2024-05-01"請寫成2024/05/01


請注意"類別"指的是消費類別，可能的"類別"清單如下:
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


時間區間部分, 請依據以下規則來計算並返回日期描述所對應的時間區間：

1. 每週的第一天是星期日。

2. 如果日期描述不清楚，例如「去年星期二」，請返回去年同月中最接近的對應星期。例如，如果今天是2024年5月1日（星期三），那麼「去年星期二」應該是2023年4月25日。

3. 「上個」或「去」表示前一個完整的週期。例如：
   - 「上個月」是指2024年4月1日至2024年4月30日。
   - 「去年」是指2023年1月1日至2023年12月31日。

4. 計算過去的時間區間時，例如「過去三個月」：
   - 假設今天是2024年4月20日，「過去三個月」指的是2024年2月1日至2024年4月20日。

5. 「過去兩週」的計算方式：
   - 假設今天是2024年5月1日，「過去兩周」指的是2024年4月21日至2024年5月1日。

6. 「本週」的日期範圍：
   - 假設今天是2024年5月1日（星期三），「本週」的日期區間是從2024年4月28日（週日）到2024年5月1日（週三），end_date 應包含當天日期。

7. 「前天」指的是兩天前的日期，start_date 和 end_date 都是該日期。

8. 「去年的這一週」或類似描述的計算方式：
   - 首先確定今年這一週的日期範圍，從上個週日開始，到當天結束。
   - 然後將這個日期範圍映射到去年，找出去年相同週日開始到該週六結束的日期範圍。





時間請依照今日是2024-05-01進行換算

用戶問題: {question}

命名實體:
"""


prompt = ChatPromptTemplate.from_template(template_2)

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

chain = prompt | model | JsonOutputParser()

# %%

csv_file = "test/test_data.csv"
output_file = "test/batch_all.json"

d = pd.read_csv(csv_file)
test_sql = d["變形問題"].values
test_sql_category = d["編號"].values

# 進行批量處理，100筆為一組
batch_size = 100

# 嘗試讀取進度檔案，若不存在則從頭開始
progress_file = "progress.pkl"
if os.path.exists(progress_file):
    with open(progress_file, "rb") as f:
        start_index = pickle.load(f)
else:
    start_index = 0

responses = []

# 從上次進度開始處理
for i in tqdm(range(start_index, len(test_sql), batch_size)):
    batch = test_sql[i : i + batch_size]
    batch_categories = test_sql_category[i : i + batch_size]
    batch_inputs = [{"today": "2024-05-01", "question": q} for q in batch]
    batch_responses = chain.batch(batch_inputs)

    # 將編號、問題和回應結合
    batch_all = [
        {"category": cat, "question": q["question"], "response": r}
        for cat, q, r in zip(batch_categories, batch_inputs, batch_responses)
    ]

    # 保存當前批次結果到JSON文件
    with open(f"test/batch_{i//batch_size + 1}.json", "w", encoding="utf-8") as f:
        json.dump(batch_all, f, ensure_ascii=False, indent=4)

    # 保存進度
    with open(progress_file, "wb") as f:
        pickle.dump(i + batch_size, f)

    responses.extend(batch_all)

# 在所有批次處理完成後，合併所有 JSON 文件成為一個大 JSON 文件
all_responses = []
for i in range(0, len(test_sql), batch_size):
    with open(f"test/batch_{i//batch_size + 1}.json", "r", encoding="utf-8") as f:
        batch_responses = json.load(f)
        all_responses.extend(batch_responses)

# 保存合併後的大 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=4)

# 清除進度檔案
if os.path.exists(progress_file):
    os.remove(progress_file)


# %%
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm




def format_date(date_str):
    # 使用 datetime 解析日期字符串
    from datetime import datetime
    try:
        # 将日期字符串转换为 datetime 对象
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
        # 将 datetime 对象格式化为 'YYYY/MM/DD' 格式的字符串
        formatted_date = date_obj.strftime('%Y/%m/%d')
        return formatted_date
    except ValueError:
        return "Invalid date format"


def nan_to_v(x):
    return '_' if pd.isna(x) else x

def none_to_v(x):
    return '_' if pd.isna(x) else x


csv_file = "test/test_data.csv"


d_ori = pd.read_csv(csv_file)
d_ori['start_date'] = d_ori['start_date'].apply(format_date)
d_ori['end_date'] = d_ori['end_date'].apply(format_date)
d_ori['store_name'] = d_ori['store_name'].apply(nan_to_v)
d_ori['category'] = d_ori['category'].apply(nan_to_v)



output_file = "test/batch_all.json"


with open(output_file, "r") as f:
    data = json.load(f)


# %%
df = pd.DataFrame(
    columns=[
        "modify_query",
        "&start_date",
        "&end_date",
        "&string",
        "&string2",
        "&string",
    ]
)


# %%
r = []
for C in list("ABCD"):
    d = [ix for ix in data if ix["category"] == C]
    d_response = pd.DataFrame(d)["response"]
    r.append(
        pd.DataFrame(
            {
                "category_1": [ix["category"] for ix in d],
                "question": [ix["question"] for ix in d],
                "modify_query": d_response.apply(lambda x: x.get("modify_query")),
                "&start_date": d_response.apply(lambda x: x.get("&start_date")),
                "&end_date": d_response.apply(lambda x: x.get("&end_date")),
                "&string1": d_response.apply(lambda x: x.get("&string1")),
                "&string2": d_response.apply(lambda x: x.get("&string2")),
                "&string": d_response.apply(lambda x: x.get("&string")),
            }
        )
    )
# %%
tot_d = pd.concat(r, axis=0)
tot_d['&string1'] = tot_d['&string1'].apply(nan_to_v)
tot_d['&string2'] = tot_d['&string2'].apply(nan_to_v)
tot_d['&string'] = tot_d['&string'].apply(nan_to_v)
# %%
ddd = pd.merge(d_ori,tot_d, left_on='變形問題',right_on='question', how = 'inner')

# %%
# ddd['_問題類別'] = (ddd['編號'] == ddd['category_1']).astype('int')
ddd['_start_date'] = (ddd['start_date'] == ddd['&start_date']).astype('int')
ddd['_end_date'] = (ddd['end_date'] == ddd['&end_date']).astype('int')
ddd['_date'] = (ddd['_start_date'] & ddd['_end_date']).astype('int')
ddd['_store_name'] = (ddd['store_name'] == ddd['&string1'] ).astype('int')
ddd['_category'] = (ddd['category'] == ddd['&string'] ).astype('int')

# %%
ddd.groupby("編號").agg(
    {
        # "_問題類別": "mean",
        "_start_date": "mean",
        "_end_date": "mean",
        "_date": "mean",
        "_start_date": "mean",
        "_store_name": "mean",
        "_category": "mean",
    }
)

# %%
ddd.to_excel('正確率計算.xlsx')
# %%


# %%
