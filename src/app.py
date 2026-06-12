import os
import streamlit as st
from model import ChatModel
import rag_util

FILES_DIR = "/content/drive/MyDrive/LLM_RAG_Bot/files"

st.title("Gemma 2B Chatbot")

@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    return model

@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    return encoder

model = load_model()

encoder = load_encoder()

def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

with st.sidebar:
  max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
  k = st.number_input("k", 1, 10, 3)
  uploaded_files = st.file_uploader("Upload PDFs for model context", type = ['pdf','PDF'], accept_multiple_files = True)

  file_paths = []
  for file in uploaded_files:
    file_paths.append(save_file(file))
  
  if uploaded_files != []:
    # create DB 
    docs = rag_util.load_and_split_pdfs(file_paths)
    DB = rag_util.FaissDb(docs = docs, embedding_function = encoder.embedding_function)

# initialise chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# display chat history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])

# accept user input
if prompt := st.chat_input('Ask me anything'):
  # add user message to chat history
  st.session_state.messages.append({'role':'user','content':prompt})
  # display user message
  with st.chat_message('user'):
    st.markdown(prompt)
  
  # display assistant response
  with st.chat_message('assistant'):
    user_prompt = st.session_state.messages[-1]['content']

    context = (None if uploaded_files == [] else DB.similarity_search(user_prompt, k = k))

    answer = model.inference(user_prompt, context = context, max_new_tokens = max_new_tokens)

    response = st.write(answer)
  
  st.session_state.messages.append({'role':'assistant','content':answer})
