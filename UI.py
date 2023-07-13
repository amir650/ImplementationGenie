import os
import glob
from dotenv import load_dotenv
import streamlit as st
from langchain import OpenAI
from llama_index import LLMPredictor, GPTVectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.chat_engine.types import ChatMode

load_dotenv()


def build_knowledge_index(directory_path):
    secret_key = os.getenv('OPENAI_API_KEY')
    print(f'building knowledge base against files defined in : {directory_path}!')
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


index = build_knowledge_index('data')
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_QUESTION,
    verbose=True
)

# Streamlit interface
st.title('Simple Conversational Interface')

# Layout for Knowledge Center and Chat
col1, col2 = st.columns(2)

# Knowledge Center Section
with col1:
    st.header('Knowledge Center')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        with open(os.path.join('data/', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File : {uploaded_file.name} has been successfully uploaded to data/ folder.  Rebuilding index ...")
        index = build_knowledge_index('data')

    if st.button("Reset Knowledge"):
        files = glob.glob('data/*')
        for f in files:
            os.remove(f)
        st.success("Knowledge has been reset!")

# Conversation Section
with col2:
    st.header('Conversational Interface')
    with st.form(key='my_form'):
        user_input = st.text_input(label=" ", key='user_input', value='', help='Type your message here')
        submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            # Bot responds to user's text
            response = chat_engine.chat(user_input)
            print(f'{response}', flush=True)

            # Input bubble in gray
            st.markdown(
                f'<div style="background-color: lightgray; padding: 10px; border-radius: 10px;">{user_input}</div>',
                unsafe_allow_html=True)

            # Response bubble in orange
            st.markdown(
                f'<div style="background-color: orange; color: white; padding: 10px; border-radius: 10px;">{response}</div>',
                unsafe_allow_html=True)

            # Clear the input field after form submission
            user_input = ''  # Clear the input by setting it to an empty string
