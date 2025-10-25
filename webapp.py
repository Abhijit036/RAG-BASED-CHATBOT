# Import the required libraries:

import os 
import google.generativeai as genai
from langchain_community.vectorstores import FAISS # This is a Vector Database.
from langchain_community.embeddings import HuggingFaceEmbeddings  # This is used to convert text to vectors.
from langchain.text_splitter import RecursiveCharacterTextSplitter # This is used to split the text into
from pypdf import PdfReader # This is used to read the PDF files.
import faiss # This is used to create the vector database.
import streamlit as st  # This is used to create the web app.
from pdfextractor import text_extractor_pdf # This is used to extract text from PDF files.



# Create MAIN page 
st.title(":green[RAG BASED CHATBOT ]")
tips = ''' Followw the steps to use this application:
* Upload your PDF file in the sidebar.
* Write your QUERY, and start chatting with the BOT.'''
st.write(tips)

# Load pdf in Side bar 
st.sidebar.title(":orange[UPLOAD YOUR PDF HERE (PDF Only)]")
file_uploaded = st.sidebar.file_uploader("Choose a file")


if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

    #STEP 1 : Configure the Models & API Key.
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    # This is the LLM model.
    llm_model = genai.GenerativeModel("gemini-2.5-flash-lite") 
    # This is the Embedding model.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

    #STEP 2 : CHUNKING.
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap =200)     
    chunks = splitter.split_text(file_text)

    # STEP 3 : Create the FIASS Vector Store.
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # STEP 4 : Configure the Retriever.
    retriever = vector_store.as_retriever(search_kwargs={"k":3}) 

    #STEP 5 : User Query.
    # Lets create a finction that takes query and generates text.
    def generate_response(query):
        # RETRIEVAL --> R
        retrieved_docs = retriever.get_relevant_documents(query=query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # ARGUMENTED PROMPT --> A
        prompt = f'''
        You are a helpful assistant using RAG,
        Here is the Context: {context}

        The Query asked by the uses is : {query}
        '''

        # GENERATION --> G
        content = (llm_model.generate_content(prompt))
        return content.text

    # Lets create a chatbot in order to start the conversation
    # Initialize chat if there is no history.
    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.markdown(f"*USER:* {msg['text']}")
        else:
            st.markdown(f"*CHATBOT:* {msg['text']}")
    
    # Input from the user - using streamlit form
    with st.form('Chat_Form', clear_on_submit=True):
        user_input = st.text_input("Enter your text here : ")
        send = st.form_submit_button("SEND")

    # Start the conversation and append the output and input-query to the history.
    if user_input and send:
        st.session_state.history.append({"role": "user", "text": user_input})

        output = generate_response(user_input)
        st.session_state.history.append({"role": "chatbot", "text": output})

        st.rerun()




    # STEP 5 : USER QUERY.
    query = st.text_input("Enter your Query here : ")
