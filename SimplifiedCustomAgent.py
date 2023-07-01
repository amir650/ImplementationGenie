from dotenv import load_dotenv
import os
from langchain import OpenAI
from llama_index import LLMPredictor, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.chat_engine.types import ChatMode

load_dotenv()


def build_knowledge_index(directory_path):
    secret_key = os.getenv('OPENAI_API_KEY')
    print(f'api key = {secret_key}')
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7,
                                            model_name="gpt-3.5-turbo",
                                            max_tokens=num_outputs,
                                            openai_api_key=secret_key))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    docs = SimpleDirectoryReader(directory_path).load_data()
    idx = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    idx.storage_context.persist()

    return idx


index = build_knowledge_index('knowledge')
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_QUESTION,
    verbose=True
)
#query_engine = index.as_query_engine()

import streamlit as st


col1, col2, col3 = st.columns([1, 2, 1])
col1.markdown(' # Welcome to my app! ')
col1.markdown(" Here is some info on the app.  ")

col2.file_uploader("upload a knowledge file")

#st.set_page_config(page_title='Ava Hoozy', page_icon=':tada:', layout='wide')
#st.subheader('hi bros')
#st.title("I'm gonna implement the shit out of you")

while True:
    question = input('>')
    response = chat_engine.chat(question)
    print(f'{response}', flush=True)
