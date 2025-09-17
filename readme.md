 InsightBot – Document-based QA System

InsightBot is an intelligent question-answering system that extracts concise answers from uploaded documents using an Agent-based architecture with MCP (Message Communication Protocol) integration.

Features
 Upload multiple document types: PDF, DOCX, PPTX, CSV, TXT, Markdown
 Clean and preprocess document content automatically
 Retrieve relevant context using FAISS vector store
 Structured MCP message passing between agents
 Generate deterministic, concise answers using LLM (EleutherAI/gpt-neo-125M)


Tech Stack Used
 Python
 Streamlit (UI)
 LangChain
 FAISS Vector Store
 HuggingFace Transformers
 EleutherAI/gpt-neo-125M Model

---
 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/mathukumalli-Sreecharan/InsightBot-QA.git
   cd InsightBot-QA
