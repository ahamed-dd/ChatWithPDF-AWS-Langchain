import json
import os
import boto3
import streamlit as st
import numpy as np
from datetime import datetime

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


bedrock = boto3.client(service_name="bedrock-runtime")
bedrockembedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
FAISS_INDEX_PATH = "faiss_index"
pdf_files_path = "pdf_files"

os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
os.makedirs(pdf_files_path, exist_ok=True)


def save_to_s3(s3_key, s3_bucket, body):
    s3 = boto3.client("s3")

    try:
        s3.put_object(Key= s3_key, Bucket=s3_bucket, Body=body)
        print("Saved to Bucket")
        return True

    except Exception as e:
        print("S3 error ", e)
        return False
    
def pdf_uploader(file):
    current_time = datetime.now().strftime("%H%M%S")
    s3_key = f"pdf/{current_time}.pdf"
    s3_bucket = "awscourse111"
    save_to_s3(s3_key, s3_bucket, file.getvalue())

def data_ingestion():
    s3 = boto3.client("s3")
    bucket_name = "awscourse111"
    response = s3.list_objects_v2(Bucket=bucket_name)

    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.lower().endswith(".pdf"):  
                local_path = os.path.join(pdf_files_path, os.path.basename(key))
                s3.download_file(bucket_name, key, local_path)
                print(f"Downloaded {key} -> {local_path}")

    loader=PyPDFDirectoryLoader(pdf_files_path)
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    get_vector = FAISS.from_documents(
        docs,
        bedrockembedding,

    )
    get_vector.save_local(FAISS_INDEX_PATH)

def get_llama():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':512})

    return llm

prompt = '''

Human: Generate 250 words for the summary description of the question, 
but give concise answer at the end.
If you do not know the answer, Tell I don't know don't make up answer

<context>
{context}
</context>

question = {question}

Assistant:

'''
PROMPT = PromptTemplate(
    template=prompt, input_variables=["context", 'question']
)

def get_response(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type= "stuff",
        retriever= vectorstore.as_retriever(
            search_type = "similarity", search_kwargs = {"k": 3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt": PROMPT})
    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat With PDF")

    user_question = st.text_input("Ask Question from the PDF files")

    with st.sidebar:
        st.info("You can either upload PDFs here to S3 (they’ll be downloaded back to your local 'pdf_files' folder automatically), or place files manually in the 'pdf_files' directory.")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Save PDF to S3"):
                with st.spinner("Uploading to S3..."):
                    pdf_uploader(uploaded_file)
                    st.success("Uploaded to S3")
        st.info("Click Vector Update to get the faiss index of the pdf files.")
        if st.button("Vector Update"):
            with st.spinner("Processing"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(FAISS_INDEX_PATH, bedrockembedding, allow_dangerous_deserialization=True)
            llm = get_llama()
            response = get_response(llm, faiss_index, user_question)
            st.write(response)
            st.success("Done")



if __name__ == "__main__":
    main()







