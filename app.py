import os
import tempfile
import re
import streamlit as st
from dataclasses import dataclass
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

# MCPMessage for structured communication
@dataclass
class MCPMessage:
    sender: str
    receiver: str
    payload: str

# Helper: clean document content
def clean_context(text: str) -> str:
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        if re.search(r"\b(what|how|why|difference|explain|define)\b", line_lower):
            continue
        if line_lower.startswith(("question:", "answer:")):
            continue
        if not line.strip():
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

# ----------------------------
# Agents
# ----------------------------

class IngestionAgent:
    def load_documents(self, uploaded_files) -> List:
        all_documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            ext = uploaded_file.name.split('.')[-1].lower()
            try:
                if ext == "pdf":
                    loader = PyPDFLoader(tmp_file_path)
                elif ext == "docx":
                    loader = UnstructuredWordDocumentLoader(tmp_file_path)
                elif ext == "csv":
                    loader = CSVLoader(tmp_file_path, encoding="utf-8")
                elif ext == "pptx":
                    loader = UnstructuredPowerPointLoader(tmp_file_path)
                elif ext in ["txt", "md"]:
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.page_content = clean_context(doc.page_content)
                all_documents.extend(docs)
            except Exception as e:
                st.error(f"Failed to load {uploaded_file.name}: {e}")

        return all_documents

class RetrievalAgent:
    def __init__(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.docs = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_index = FAISS.from_documents(self.docs, embeddings)

    def retrieve(self, query, k=3):
        retriever = self.vector_index.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)

class LLMResponseAgent:
    def __init__(self, llm_pipeline):
        self.llm_pipeline = llm_pipeline

    def generate_answer(self, user_query, retrieved_docs):
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
You are an expert. Answer the question concisely based only on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{user_query}

Answer concisely:
"""

        result = self.llm_pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.0,            # Deterministic output
            top_p=0.95,
            repetition_penalty=1.5,     # Stronger penalty to prevent repeats
            do_sample=False,            # Greedy decoding
        )

        if isinstance(result, list) and isinstance(result[0], dict):
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = str(result)

        # Remove prompt echo if present
        generated_text = generated_text.replace(prompt, "").strip()

        # Truncate at first double newline or repetition artifact
        generated_text = re.split(r'\n{2,}|Apple’s brand identity is associated', generated_text)[0].strip()

        return MCPMessage(sender="LLMResponseAgent", receiver="User", payload=generated_text)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Multi-Agent QA Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Insight bot")
st.write("Upload documents (PDF, DOCX, PPTX, CSV, TXT/Markdown) and ask questions.")

uploaded_files = st.file_uploader(
    "Upload documents", type=["pdf", "docx", "pptx", "csv", "txt", "md"], accept_multiple_files=True
)

if uploaded_files:
    ingestion_agent = IngestionAgent()
    all_docs = ingestion_agent.load_documents(uploaded_files)
    st.success(f"✅ {len(all_docs)} documents loaded successfully!")

    retrieval_agent = RetrievalAgent(all_docs)

    # Initialize LLM (CPU-safe)
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm_agent = LLMResponseAgent(llm_pipeline)

    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer") and user_query.strip():
        retrieved_docs = retrieval_agent.retrieve(user_query)
        answer_message = llm_agent.generate_answer(user_query, retrieved_docs)
        st.markdown("**Answer:**")
        st.write(answer_message.payload)
