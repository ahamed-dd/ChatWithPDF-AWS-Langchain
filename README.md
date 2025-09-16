# Chat With PDF (RAG App)

This project is a **Retrieval-Augmented Generation (RAG) application** built with **Streamlit, AWS S3, FAISS, and LangChain**.  
It allows you to upload PDF files, store them in S3, build a FAISS vector index, and ask natural language questions based on the content of the PDFs.

---

## Features
Upload PDF files via Streamlit sidebar
Store and retrieve files from **AWS S3**
Process multiple PDF files automatically
Create embeddings using **Bedrock Embeddings**
Build a FAISS vector database
Ask questions and get answers directly from your PDF content

---

## Project Structure

├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── faiss_index/ # Stores FAISS index files
└── pdf_files/ # Stores local PDF files

To run the app:
streamlit run app.py
