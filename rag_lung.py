import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load Llama Model
def load_llama_model():
    model_path = "/home/h392x566/llama3.2-8b-train-py"
    return pipeline("text-generation", model=model_path, device=0)  # Use GPU if available

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create Vector Store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load Vector Store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Create Conversational Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = HuggingFacePipeline(pipeline=load_llama_model())
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle User Questions
def user_input(user_question):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main Application
def main():
    st.set_page_config("AI Lung Cancer Oncology Assistant", page_icon=":scroll:")
    st.header("AI Lung Cancer RAG Oncology Assistant 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/xuan13hao" target="_blank">Hao Xuan</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
