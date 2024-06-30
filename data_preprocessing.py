# %%
import itertools
import pandas as pd

qa = pd.read_excel("./cubelab.xlsx", sheet_name="QA")
data_time = pd.read_excel("./cubelab.xlsx", sheet_name="時間")
data_name = pd.read_excel("./cubelab.xlsx", sheet_name="特店名稱")
data_category = pd.read_excel("./cubelab.xlsx", sheet_name="消費類別")

# %%
list_category = list(data_category["消費類別"])
list_name = list(data_name["特店名稱"])
list_time = list(data_time["替代變數"])

# %%
l = []

# %%
for t in list_time:
    d = qa[qa["編號"] == "A"]["變形問題"].apply(
        lambda x: x.replace("[某個消費時間]", str(t))
    )
    l.append(
        pd.DataFrame(
            {
                "category": "A",
                "Q": d,
            }
        )
    )
# %%
for t, n in itertools.product(list_time, list_name):

    d = qa[qa["編號"] == "B"]["變形問題"].apply(
        lambda x: x.replace("[某個消費時間]", str(t)).replace("[某個商戶]", str(n))
    )
    l.append(
        pd.DataFrame(
            {
                "category": "B",
                "Q": d,
            }
        )
    )
# %%
for t, c in itertools.product(list_time, list_category):

    d = qa[qa["編號"] == "C"]["變形問題"].apply(
        lambda x: x.replace("[某個消費時間]", str(t)).replace("[消費類別]", str(c))
    )
    l.append(
        pd.DataFrame(
            {
                "category": "C",
                "Q": d,
            }
        )
    )
# %%
for t in list_time:
    d = qa[qa["編號"] == "D"]["變形問題"].apply(
        lambda x: x.replace("[時間]", str(t))
    )
    l.append(
        pd.DataFrame(
            {
                "category": "D",
                "Q": d,
            }
        )
    )
# %%
df = pd.concat(l, axis=0)
df.reset_index(drop=True, inplace=True)
a_df = (
    qa[["編號", "問題類別", "SQL1", "SQL2", "SQL3"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
a_df.rename({"編號": "category"}, axis=1, inplace=True)
# %%
a_df["SQL1"] = a_df["SQL1"].apply(lambda x: x.replace("\n", " "))
a_df["SQL2"] = a_df["SQL2"].apply(lambda x: x.replace("\n", " "))
a_df["SQL3"] = a_df["SQL3"].apply(lambda x: x.replace("\n", " "))

# %%
df = df.merge(a_df, on="category", how="left")

# %%
df.rename({"Q": "變形問題"}, axis=1, inplace=True)
# %%
df.to_csv("qa_set_with_sql.csv", index=False)
# %%

# %%
