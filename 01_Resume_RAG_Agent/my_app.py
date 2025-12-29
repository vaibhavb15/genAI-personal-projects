import streamlit as st
import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="Resume Chatbot", page_icon="🤖")
st.title("🤖 Chat With My Resume")

# Load the key securely from streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing Google API Key. Please add it to .streamlit/secrets.toml")
    st.stop()

# --- CACHED FUNCTIONS (The "Engine") ---
# We use @st.cache_resource so these only run ONCE, not every time you ask a question.

@st.cache_resource
def load_and_process_resume():
    """Loads the PDF and creates the vector index."""
    file_path = "my_resume.pdf"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}. Please put your resume in the same folder!")
        return None
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Use local embeddings to save money/quota
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

@st.cache_resource
def setup_rag_chain(_vectorstore):
    # --- OPTIMIZATION 1: INCREASE RETRIEVAL (k) ---
    # By default, k=4. We increase it to 6 to capture more parts of your resume.
    # We also use 'mmr' (Max Marginal Relevance) to get DIVERSE chunks, not just repetitive ones.
    retriever = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    
    # --- OPTIMIZATION 2: BETTER MODEL PARAMETERS ---
    # We keep temperature low for facts, but not zero so it sounds natural.
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)
    
    # --- OPTIMIZATION 3: BETTER PROMPT ENGINEERING ---
    # We removed the length constraint and gave it a "Persona".
    system_prompt = (
        "You are a senior recruiter and career coach representing the candidate. "
        "Your goal is to highlight the candidate's strengths based strictly on the context provided. "
        "\n\n"
        "Guidelines:"
        "\n1. Answer directly and professionally."
        "\n2. If the user asks about a specific skill, cite the specific job or project where it was used."
        "\n3. Do not invent information. If the answer is not in the resume, say: 'The resume does not explicitly mention this.'"
        "\n4. Use bullet points for lists to make it readable."
        "\n\n"
        "Context from Resume:"
        "\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# --- UI LOGIC ---

# 1. Load data
with st.spinner("Loading resume..."):
    vectorstore = load_and_process_resume()

if vectorstore:
    rag_chain = setup_rag_chain(vectorstore)

    # 2. Chat Interface
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle User Input
    if prompt := st.chat_input("Ask a question about my experience..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response['answer']
                st.markdown(answer)
        
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})