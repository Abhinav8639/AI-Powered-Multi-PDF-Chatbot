import streamlit as st
import os
import traceback
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please add it and restart the app.")
    st.stop()

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# ───────────────────────────────────────────────
#          PDF TEXT EXTRACTION
# ───────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text.strip()

# ───────────────────────────────────────────────
#          TEXT CHUNKING
# ───────────────────────────────────────────────
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

# ───────────────────────────────────────────────
#          CREATE / SAVE VECTOR STORE
# ───────────────────────────────────────────────
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

# ───────────────────────────────────────────────
#          BUILD LCEL RETRIEVAL CHAIN
# ───────────────────────────────────────────────
def build_chain():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_output_tokens=2048
        )

        prompt = PromptTemplate.from_template(
            """
            You are a helpful document Q&A assistant.
            Answer the question using ONLY the provided context.
            If the information is not in the context, say:
            "Sorry, the answer is not available in the uploaded documents."

            Context:
            {context}

            Question: {question}

            Answer in a clear, concise and natural way:
            """
        )

        chain = (
            RunnableParallel(
                {
                    "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                    "question": RunnablePassthrough()
                }
            )
            | prompt
            | llm
        )

        return chain

    except Exception as e:
        st.error(f"Error building chain: {str(e)}")
        return None

# ───────────────────────────────────────────────
#          HANDLE USER QUESTION
# ───────────────────────────────────────────────
def user_query_handler(question):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDFs first.")
        return

    chain = build_chain()
    if chain is None:
        return

    with st.spinner("Thinking... (free tier can take 10–120s; may hang if quota hit)"):
        try:
            # Timeout after 90 seconds to avoid infinite hang
            response = chain.invoke(question, config={"timeout": 90})
            st.markdown("### 🤖 Answer:")
            st.markdown(response.content)
        except Exception as e:
            st.error(f"**Generation failed:** {str(e)}")
            st.error("Full traceback (for debugging):")
            st.exception(e)  # Shows complete error stack in UI
            st.info("""
            **Common fixes:**
            - Free tier quota/rate limit hit → wait 5–10 min or until tomorrow ~1:30 PM IST
            - Billing account not linked → add one at https://console.cloud.google.com/billing (no charge under free limits)
            - API key issue → create new key at https://aistudio.google.com/app/apikey
            - Try the 'Test Direct Gemini' button in sidebar first
            """)

# ───────────────────────────────────────────────
#          MAIN STREAMLIT APP
# ───────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Multi PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )

    st.header("📚 Multi-PDF Chat Agent (Gemini 1.5 Flash)")

    user_question = st.text_input(
        "Ask anything about your uploaded PDFs:",
        key="question_input"
    )

    if user_question:
        user_query_handler(user_question)

    with st.sidebar:
        st.image("img/Robot.jpg", width="stretch")
        st.title("📁 Upload Documents")

        pdf_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDFs at once"
        )

        if st.button("Process PDFs", type="primary"):
            if pdf_files:
                with st.spinner("Extracting text..."):
                    raw_text = get_pdf_text(pdf_files)
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs.")
                        return

                with st.spinner("Creating chunks & embeddings... (this may take a minute)"):
                    chunks = get_text_chunks(raw_text)
                    success = get_vector_store(chunks)
                    if success:
                        st.success(f"Processed {len(chunks)} chunks successfully! Ready to chat.")
            else:
                st.warning("Please upload at least one PDF file.")

        # ─── Debug Button: Test LLM directly (bypass retrieval) ───
        if st.button("Test Direct Gemini (no PDF)"):
            try:
                test_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.0
                )
                with st.spinner("Direct test (should be fast)..."):
                    resp = test_llm.invoke("Hello! What is the current year and your model name?")
                    st.success("Direct response: " + resp.content)
            except Exception as e:
                st.error(f"Direct test failed: {str(e)}")
                st.exception(e)

        st.divider()
        st.image("img/gkj.jpg", width=180)
        st.caption("AI App created by Gurpreet Kaur 😊")

    st.markdown(
        """
        <hr style="margin-top: 60px;">
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            © 2026 Multi-PDF Chatbot • Powered by Google Gemini
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()