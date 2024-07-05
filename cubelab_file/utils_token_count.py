# %%
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
import tiktoken

# %%
model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)
# %%
message = HumanMessage(
    content="Translate this sentence from English to French. I love programming."
)
# %%
with get_openai_callback() as cb:
    model.invoke([message])
print(f"Total Token: {cb.total_tokens:.0f}")  
print(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")  

# %%
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(f"Total Token: {num_tokens_from_string('tiktoken is great!')}")  
# %%
