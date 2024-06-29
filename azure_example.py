
# %%
import os
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
# %%
message = HumanMessage(
    content="Translate this sentence from English to French. I love programming."
)
model.invoke([message])
# %%
from langchain.callbacks import get_openai_callback
with get_openai_callback() as cb:
    model.invoke([message])
    print(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")  

# %%

from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    # openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    
)
# %%
text = "this is a test document"

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])

doc_result[0][:5]

# %%