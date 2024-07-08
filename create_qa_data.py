import itertools
import pandas as pd
qa = pd.read_excel("../cubelab_mod.xlsx", sheet_name="QA")
data_time = pd.read_excel("../cubelab_mod.xlsx", sheet_name="時間")
data_name = pd.read_excel("../cubelab_mod.xlsx", sheet_name="特店名稱")
data_問題類別 = pd.read_excel("../cubelab_mod.xlsx", sheet_name="消費類別")

# %%
list_問題類別 = list(data_問題類別["消費類別"])
list_name = list(data_name["特店名稱"])
list_time = list(data_time["替代變數"])

# %%
l = []

# %%
for t in list_time:
    d = qa[qa["編號"] == "A"]["變形問題"].apply(
        lambda x: x.replace("[時間]", str(t))
    )
    l.append(
        pd.DataFrame(
            {
                "問題編號": "A",
                "變形問題": qa[qa["編號"] == "A"]["變形問題"],
                "變形問題_填空": d,
                "時間": t,
                "商戶": None,
                "類別": None,
            }
        )
    )
# %%
for t, n in itertools.product(list_time, list_name):

    d = qa[qa["編號"] == "B"]["變形問題"].apply(
        lambda x: x.replace("[時間]", str(t)).replace("[商戶]", str(n))
    )
    l.append(
        pd.DataFrame(
            {
                "問題編號": "B",
                "變形問題": qa[qa["編號"] == "B"]["變形問題"],
                "變形問題_填空": d,
                "時間": t,
                "商戶": n,
                "類別": None,
            }
        )
    )
# %%
for t, c in itertools.product(list_time, list_問題類別):

    d = qa[qa["編號"] == "C"]["變形問題"].apply(
        lambda x: x.replace("[時間]", str(t)).replace("[類別]", str(c))
    )
    l.append(
        pd.DataFrame(
            {
                "問題編號": "C",
                "變形問題": qa[qa["編號"] == "C"]["變形問題"],
                "變形問題_填空": d,
                "時間": t,
                "商戶": None,
                "類別": c,
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
                "問題編號": "D",
                "變形問題": qa[qa["編號"] == "D"]["變形問題"],
                "變形問題_填空": d,
                "時間": t,
                "商戶": None,
                "類別": None,
            }
        )
    )
# %%
df = pd.concat(l, axis=0)
# breakpoint()
df.reset_index(drop=True, inplace=True)
a_df = (
    qa[["變形問題", "SQL1", "SQL2", "SQL3"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
# %%
a_df["SQL1"] = a_df["SQL1"].apply(lambda x: x.replace("\n", " "))
a_df["SQL2"] = a_df["SQL2"].apply(lambda x: x.replace("\n", " "))
a_df["SQL3"] = a_df["SQL3"].apply(lambda x: x.replace("\n", " "))

# %%
df = df.merge(a_df, on="變形問題", how="left")

# # %%
df.to_csv("qa_set_with_sql_mod.csv", index=False)