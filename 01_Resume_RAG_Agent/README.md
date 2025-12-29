\# 🤖 AI Resume Agent (RAG)



An autonomous AI agent that answers questions about my professional background using Retrieval Augmented Generation (RAG).



\*\*Built with:\*\* Python, LangChain, Google Gemini (Flash-Lite), FAISS, and Streamlit.



\### 🎯 Project Goal

To demonstrate how Generative AI can be applied to unstructured data (like resumes) to create interactive, fact-based user experiences. This project solves the "hallucination" problem by grounding the LLM in a specific knowledge base (my resume).



\### ⚙️ Technical Architecture

1\.  \*\*Ingestion:\*\* Loads PDF documents and splits them into semantic chunks using `RecursiveCharacterTextSplitter`.

2\.  \*\*Embedding:\*\* Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) for cost-efficient vectorization.

3\.  \*\*Vector Store:\*\* Indexes chunks in a local FAISS vector database.

4\.  \*\*Retrieval:\*\* Uses MMR (Maximal Marginal Relevance) search to find diverse, relevant context.

5\.  \*\*Generation:\*\* Powered by Google's \*\*Gemini Flash-Lite\*\* model for low-latency inference.



\### 🚀 How to Run Locally



\*\*1. Clone the repository\*\*

```bash

git clone \[https://github.com/vaibhavb15/genAI-personal-projects.git](https://github.com/vaibhavb15/genAI-personal-projects.git)

cd genAI-personal-projects/01\_Resume\_RAG\_Agent



```



\*\*2. Install dependencies\*\*



```bash

pip install -r requirements.txt



```



\*\*3. Set up API Keys\*\*

This project uses Google Gemini. You need your own API key to run it.



\* Create a folder named `.streamlit` in the root directory.

\* Inside that folder, create a file named `secrets.toml`.

\* Add your key to the file like this:

```toml

GOOGLE\_API\_KEY = "Your\_Google\_API\_Key\_Here"



```





\*\*4. Run the App\*\*



```bash

streamlit run my\_app.py



```

