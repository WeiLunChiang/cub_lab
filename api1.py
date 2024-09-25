import os
from uuid import uuid4
from typing import Dict
from functools import partial
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from utils_retriever import RetrieveWithScore, get_metadata_runnable
from utils_vectordb import get_or_create_http_chromadb
from constant import PROMPT_COMPLETETION, PROMPT_QUESTION_NER
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)


class ChainNer:
    def __init__(
        self,
        sessionId: str,
        customerId: str,
        chromaHost: str,
        chromaPort: str,
        chromaCollection: str,
        sqlPath: str,
        today: str,
        k: int = 1,
        scoreThreshold: float = 0,
        **kwargs,
    ):
        self.today = self._convert_time_format(today)
        self.sessionId = sessionId
        self.customerId = customerId
        self.model = self._create_model()
        self.retriever = self._create_retriever(
            chromaHost, chromaPort, chromaCollection, k, scoreThreshold
        )
        self.chian_completeion = self._create_chain_completeion(sqlPath)
        self.chain_ner = self._create_chain_ner()

    def search(self, user_input: str, **kwargs) -> Dict:

        response = {}
        template = {}
        template["tid"] = None
        template["blockReason"] = None
        template["startDate"] = None
        template["endDate"] = None
        template["storeName"] = [None]
        template["categoryName"] = [None]
        template["message"] = None

        try:
            if "被阻擋" in (
                user_input := self.chian_completeion.invoke(
                    {"user_input": user_input},
                    config={"configurable": {"session_id": self.sessionId}},
                )
            ):

                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input
                # self.chain_ner.invoke(user_input)會失敗
            elif "被阻擋" in (result := self.chain_ner.invoke(user_input)):

                template["tid"] = "98"  # TBD
                template["blockReason"] = user_input

            elif not result["retriever"]["score"]:

                template["tid"] = "99"  # TBD
                template["blockReason"] = "相似度過低"

            else:

                template["tid"] = result["retriever"].get("category", None)[0]
                template["startDate"] = result["keys"].get("&start_date", None)
                template["endDate"] = result["keys"].get("&end_date", None)
                template["storeName"] = [
                    result["keys"].get("&string1", None),
                    result["keys"].get("&string2", None),
                ]
                template["categoryName"] = [result["keys"].get("&string", None)]
                template["message"] = user_input

        except Exception as e:
            template["tid"] = "99"
            template["blockReason"] = e

        finally:

            response["sessionId"] = self.sessionId
            response["customerId"] = self.customerId
            response["template"] = template

        return response

    @staticmethod
    def _convert_time_format(time_str):
        dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")

    def _create_model(self) -> AzureChatOpenAI:
        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        return model

    def _create_retriever(
        self,
        chromaHost: str,
        chromaPort: str,
        chromaCollection: str,
        k: int,
        scoreThreshold: float,
    ) -> "retriever":
        vectorstore = get_or_create_http_chromadb(
            host=chromaHost,
            port=chromaPort,
            collection_name=chromaCollection,
        )
        retriever = RetrieveWithScore(
            vectorstore,
            k=k,
            score_threshold=scoreThreshold,
        )
        return retriever

    def _create_chain_completeion(self, connection) -> RunnableWithMessageHistory:
        prompt = ChatPromptTemplate.from_template(PROMPT_COMPLETETION)

        chain = prompt | self.model | StrOutputParser()

        get_chat_history = partial(SQLChatMessageHistory, connection=connection)

        chian_completeion = RunnableWithMessageHistory(
            chain,
            get_chat_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )
        return chian_completeion

    def _create_chain_ner(self):

        prompt = ChatPromptTemplate.from_template(PROMPT_QUESTION_NER)
        json_parser_chain = prompt | self.model | JsonOutputParser()

        retriever_chain = (
            RunnableLambda(lambda response: response["modify_query"])
            | self.retriever
            | {  # vectordb拉到的內容(包含SQL)
                "SQL": get_metadata_runnable("SQL1", "SQL2", "SQL3"),
                "標準問題": get_metadata_runnable("問題類別"),
                "category": get_metadata_runnable("category"),
                "score": get_metadata_runnable("score"),
            }
        )

        chain_ner = (
            {
                "question": RunnablePassthrough(),
                "today": RunnableLambda(lambda x: self.today),
            }
            | json_parser_chain
            | {
                "keys": RunnablePassthrough(),  # 解析出來的參數
                "retriever": retriever_chain,
            }
        )
        return chain_ner


import sys


def main():
    sessionId = str(uuid4())
    customerId = "A"
    userInputRaw = "過去一年我在蝦皮、優食花了多少錢?"
    # userInputRaw = "HI"
    # userInputRaw = sys.argv[3]

    today = "2024/05/01 14:00:03"

    chain_ner = ChainNer(
        sessionId=sessionId,
        customerId=customerId,
        chromaHost=os.environ["CHROMA_HOST"],
        chromaPort=os.environ["CHROMA_PORT"],
        chromaCollection=os.environ["CHROMA_COLLECTION"],
        sqlPath=os.environ["SQL_PATH"],
        today=today,
    )

    response = chain_ner.search(user_input=userInputRaw)
    print(response)


if __name__ == "__main__":
    main()
