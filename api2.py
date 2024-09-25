import os
from uuid import uuid4
from typing import Dict, List
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from module2_llm_response import cust_desc_dict
from constant import PROMPT_GENAI_RESPONSE
import mlflow


class GenAIResponse:

    def __init__(self, sessionId, customerId):
        self.sessionId = sessionId
        self.customerId = customerId
        self.chain = self._create_chain_response()

    def _create_chain_response(self):

        model = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_GENAI_RESPONSE)
        parser = StrOutputParser()

        chain = prompt | model | parser
        return chain

    def generate_answer(
        self,
        tid: str,
        message: str,
        consumptionNumber: str,
        totalAmount: str,
        storeName: List,
        categoryName: List,
        **kwargs,
    ) -> Dict:

        response = {}
        response["genAI"] = {"message": None}
        response["template"] = {"tid": None, "blockReason": None}

        try:
            if "被阻擋" in (
                gen_ai_message := self.chain.invoke(
                    {
                        "message": message,
                        "consumptionNumber": consumptionNumber,
                        "totalAmount": totalAmount,
                        "storeName": storeName,
                        "categoryName": categoryName,
                        "tone": "一般用戶",
                        "desc": cust_desc_dict.get("一般用戶"),
                    }
                )
            ):
                response["template"]["tid"] = "98"
                response["template"]["blockReason"] = gen_ai_message

            else:

                response["genAI"]["message"] = gen_ai_message
                response["template"]["tid"] = tid
                response["template"]["blockReason"] = None

        except Exception as e:
            response["template"]["tid"] = "98"
            response["template"]["blockReason"] = e

        finally:
            response["sessionId"] = self.sessionId
            response["customerId"] = self.customerId

        return response


def main():
    sessionId = str(uuid4())
    customerId = "A"
    message = "我上週在蝦皮花了多少錢?"
    consumptionNumber = "50"
    totalAmount = "10000"
    storeName = "蝦皮"
    categoryName = "旅宿業"
    tid = "B"
    genai_response = GenAIResponse(
        sessionId=sessionId,
        customerId=customerId,
    )

    response = genai_response.generate_answer(
        message=message,
        tid=tid,
        consumptionNumber=consumptionNumber,
        totalAmount=totalAmount,
        storeName=storeName,
        categoryName=categoryName,
    )
    print(response)


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("cubelab_demo")
    mlflow.langchain.autolog()

    with mlflow.start_run(run_name="api2_test3") as run:
        main()
