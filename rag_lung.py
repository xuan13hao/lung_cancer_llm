import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


st.set_page_config(page_title="AI System with RAG", layout="wide")

@st.cache_resource
def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer

@st.cache_resource
def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(texts, embedding=embeddings, persist_directory="chromadb")
    vector_store.persist()

    return vector_store

@torch.inference_mode()
def generate_response(model, tokenizer, prompt, generation_config):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to("cuda")

    pad_token_id = tokenizer.pad_token_id
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=generation_config.max_new_tokens,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        do_sample=generation_config.do_sample,
        pad_token_id=pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def configure_generation_settings():
    with st.sidebar:
        st.title("Generation Settings")
        max_new_tokens = st.slider("Max New Tokens", 50, 1000, 150, step=10)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, step=0.01)
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, step=0.1)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, step=0.1)
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )

def main():
    st.title("AI System with Retrieval-Augmented Generation (RAG)")

    model_path = "/home/xuan/llama3.2-8b-train-py"
    pdf_path = st.sidebar.file_uploader("Upload PDF for Knowledge Base", type="pdf")

    st.sidebar.title("Model Settings")
    st.sidebar.text(f"Model: {model_path}")

    model, tokenizer = load_model_and_tokenizer(model_path)
    generation_config = configure_generation_settings()

    if pdf_path:
        vector_store = create_vector_store(pdf_path)
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question"],
                    template="Answer the question based on the context: {context}\nQuestion: {question}\nAnswer:"
                )
            }
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter your query:"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Thinking...")

                result = qa_chain({"question": prompt})
                response = result["answer"]
                sources = result["source_documents"]

                response_placeholder.markdown(response)
                if sources:
                    st.markdown("### Sources:")
                    for source in sources:
                        st.markdown(f"- {source.metadata['source']}")

            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
