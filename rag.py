# =============================
# RAG BASED CHATBOT (Streamlit)
# Compatible with LangChain v0.3+ and Gemini 2.5
# =============================

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import google.generativeai as genai
import streamlit as st
from pdfextractor import text_extractor_pdf
import time

# =============================
# Sidebar - Upload & Settings
# =============================
st.sidebar.title("📂 Upload your PDF file")
file_uploaded = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

st.title("💬 :green[RAG BASED CHATBOT]")
st.caption("Fully interactive SaaS-style chatbot with dynamic animations 🤖")

tips = """
✅ **Steps to use:**
1. Upload a PDF file using the sidebar.  
2. Start chatting below with your queries.  
"""
st.info(tips)

# =============================
# Initialize Session State
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# =============================
# UI Theme Colors
# =============================
if dark_mode:
    bg_color = "#1E1E1E"
    user_bubble = "#0A84FF"
    assistant_bubble = "#2C2C2C"
    text_user = "white"
    text_assistant = "white"
else:
    bg_color = "#fafafa"
    user_bubble = "#0078FF"
    assistant_bubble = "#F1F0F0"
    text_user = "white"
    text_assistant = "black"

# =============================
# Custom CSS Animations & Styles
# =============================
st.markdown(f"""
<style>
body {{
    background: linear-gradient(270deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
    background-size: 800% 800%;
    animation: gradientBG 20s ease infinite;
}}
@keyframes gradientBG {{
    0% {{background-position:0% 50%;}}
    50% {{background-position:100% 50%;}}
    100% {{background-position:0% 50%;}}
}}
.chat-box {{
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 12px;
    background-color: {bg_color};
    position: relative;
    scroll-behavior: smooth;
}}
.chat-container {{
    display: flex;
    align-items: flex-start;
    margin: 12px 0;
    opacity: 0;
    transform: translateY(20px) scale(0.8);
    animation: chatFadeIn 0.5s forwards;
}}
@keyframes chatFadeIn {{
    0% {{ opacity: 0; transform: translateY(20px) scale(0.8); }}
    80% {{ transform: translateY(-5px) scale(1.05); }}
    100% {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.user-container {{ flex-direction: row-reverse; }}
.avatar {{
    font-size: 28px;
    margin: 0 10px;
    cursor: grab;
    transition: transform 0.3s;
}}
.avatar:hover {{ transform: scale(1.2) rotate(-5deg); }}
.bubble {{
    padding: 14px 20px;
    border-radius: 20px;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.5s ease;
    position: relative;
}}
.bubble:hover {{
    transform: scale(1.03) translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}}
.user-bubble {{ background-color: {user_bubble}; color: {text_user}; }}
.assistant-bubble {{ background-color: {assistant_bubble}; color: {text_assistant}; }}
.typing-dots span {{
    display: inline-block;
    width: 6px;
    height: 6px;
    margin: 0 2px;
    background-color: {text_assistant};
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}}
.typing-dots span:nth-child(1) {{ animation-delay: 0s; }}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes bounce {{
    0%, 80%, 100% {{ transform: scale(0); }}
    40% {{ transform: scale(1); }}
}}
.pdf-highlight {{
    background-color: #fffa65;
    padding: 2px 4px;
    border-radius: 4px;
    animation: glow 1s ease-in-out infinite alternate;
}}
@keyframes glow {{
    from {{ box-shadow: 0 0 5px #fffa65; }}
    to {{ box-shadow: 0 0 20px #fffa65; }}
}}
.chat-box::-webkit-scrollbar {{
    width: 6px;
}}
.chat-box::-webkit-scrollbar-thumb {{
    background-color: rgba(0,0,0,0.2);
    border-radius: 3px;
}}
#back-to-top {{
    position: sticky;
    bottom: 10px;
    text-align: right;
}}
.back-btn {{
    padding: 5px 10px;
    background-color: {user_bubble};
    color: white;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
}}
.st-chat-input {{
    position: sticky;
    bottom: 0;
    z-index: 10;
    background-color: {bg_color};
    padding-top: 10px;
}}
</style>
""", unsafe_allow_html=True)

# =============================
# PDF Processing & Chat Logic
# =============================
if file_uploaded:
    # Extract text from uploaded PDF
    file_text = text_extractor_pdf(file_uploaded)

    if file_text.strip():
        # Configure Gemini API
        key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=key)
        llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

        # Create embeddings and chunks
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(file_text)

        if chunks:
            vector_store = FAISS.from_texts(chunks, embedding_model)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Chat input
            user_input = st.chat_input("Type your question here...")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Retrieve relevant PDF chunks
                retrieved_docs = retriever.get_relevant_documents(user_input)
                context = " ".join(
                    [f"<span class='pdf-highlight'>{doc.page_content}</span>" for doc in retrieved_docs]
                )

                # Build the LLM prompt
                prompt = f"""
                You are a helpful assistant.
                Context: {context}
                User question: {user_input}
                """

                # Typing animation placeholder
                placeholder = st.empty()
                placeholder.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="avatar">🤖</div>
                        <div class="bubble assistant-bubble">
                            <div class="typing-dots"><span></span><span></span><span></span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Generate response using Gemini
                with st.spinner("🤔 Assistant is thinking..."):
                    response = llm_model.generate_content(prompt).text
                    time.sleep(1)

                # Animated typing effect
                typed_text = ""
                for char in response:
                    typed_text += char
                    placeholder.markdown(
                        f"""
                        <div class="chat-container">
                            <div class="avatar">🤖</div>
                            <div class="bubble assistant-bubble">{typed_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.01)

                st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display chat history
            st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-container user-container">
                            <div class="avatar">👤</div>
                            <div class="bubble user-bubble">{chat['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-container">
                            <div class="avatar">🤖</div>
                            <div class="bubble assistant-bubble">{chat['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Back-to-top button
            st.markdown(
                """
                <div id='back-to-top'>
                    <span class='back-btn' onclick="document.querySelector('.chat-box').scrollTop = 0;">⬆️ Back to Top</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Smooth auto-scroll
            st.markdown(
                """
                <script>
                var chatBox = window.parent.document.querySelector('.chat-box');
                if (chatBox) {
                    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
                }
                </script>
                """,
                unsafe_allow_html=True,
            )
