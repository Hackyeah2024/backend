import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

openai_api_key = os.environ.get('OPENAI_API_KEY')

open_ai_llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_api_key)
open_ai_llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)