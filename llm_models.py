import os
from langchain_community.chat_models import ChatOpenAI, ChatCohere
from langchain.llms import Cohere
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

openai_api_key = os.environ.get('OPENAI_API_KEY')
cohere_api_key = os.environ.get('COHERE_API_KEY')

open_ai_llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=openai_api_key)
command_r_plus_llm = Cohere(model="command-r-plus", temperature=0.0, cohere_api_key=cohere_api_key)

open_ai_llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

