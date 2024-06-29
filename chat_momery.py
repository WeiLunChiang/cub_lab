# %%
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from functools import partial
from langchain_core.output_parsers import StrOutputParser

# %%

# session_id = "108"

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

template = """
你是一個智能問句補完助手，如判斷用戶的問句已經完整包含商家、時間等資訊, 則不進行修改, 直接返回原問句。
如判斷資訊不完整，你需要根據用戶的對話歷史中最後的AIMessage補充用戶當前輸入中缺失的信息，
確保問句包含完整的主受詞（如時間、商家、消費類別等）。

請注意"那"、"還是"等連接詞可能表示接續的意思, 例如那ABC店家呢? 指的是"接續前文, ABC店家呢?"

chat_history:
{history}

user_input:
{user_input}

請根據對話歷史補充用戶當前輸入中的缺失信息，只返回補完後的問句。
"""


prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm | StrOutputParser()

get_chat_history = partial(SQLChatMessageHistory, connection="sqlite:///memory.db")

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

# %%
if __name__=='__main__':
    response = runnable_with_history.invoke(
        {"user_input": "我上周在蝦皮網購的金額是多少?"},
        config={"configurable": {"session_id": session_id}},
    )
    print(response)

    # %%
    mem = get_chat_history(session_id)
# response = runnable_with_history.invoke(
#     {"user_input": "那本周呢?"},
#     config={"configurable": {"session_id": session_id}},
# )
# print(response)

# # %%
# response = runnable_with_history.invoke(
#     {"user_input": "吃飯多少次?"},
#     config={"configurable": {"session_id": session_id}},
# )
# print(response)


# # %%
# print(get_chat_history(session_id).messages)

# # %%
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain_community.chat_message_histories import SQLChatMessageHistory
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from functools import partial

# llm = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
#     openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
# )

# # template = """
# #     你的工作是把所有對話紀錄區分humanmessage、aimessage並加上編號、日期, 如果無對話紀錄則輸出'無紀錄'
# #     不用回答用戶問題, 僅進行格式輸出的工作
# # """
# # %%
# prompt = ChatPromptTemplate.from_messages(
#     [
#         # ("system", template),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}"),
#     ]
# )

# chain = prompt | llm

# # session_id = "99999"
# # history = SQLChatMessageHistory(session_id, connection="sqlite:///memory.db")


# # %%
# question = "請幫我計算對話紀錄中有幾則HUMANmessage、AIMESSAGE, 並列出對應的messages, 列出所有的結果"
# response = chain.invoke(
#     {
#         # "history": history.messages,
#         "history": mem.messages,
#         "question": question,
#     }
# )
# print(response.content)


# # %%
# mem.add_ai_message(response.content)
# mem.add_user_message(response.content)


# # %%
# history.messages

# # %%
# p = prompt.invoke(
#     {
#         "history": history.messages,
#         "question": question,
#     }
# )
# # %%
# p.messages
# # %%

# %%
