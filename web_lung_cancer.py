import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


st.set_page_config(page_title="Lung Cancer AI System", layout="wide")

@st.cache_resource
def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Automatically maps model to GPU/CPU
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


    model.eval()  
    return model, tokenizer

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
    st.title("Lung Cancer AI System")

    model_path = "/home/h392x566/llama3.2-8b-train-py"  
    st.sidebar.title("Model Settings")
    st.sidebar.text(f"Model: {model_path}")

    model, tokenizer = load_model_and_tokenizer(model_path)
    generation_config = configure_generation_settings()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Enter your query:"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add preamble/instruction to the prompt
        preamble = (
            "I am a helpful AI Lung Cancer Oncology Assistant. "
            "Provide one answer ONLY to the following query based on the context provided below. "
            "Do not generate or answer any other questions. "
            "Do not make up or infer any information that is not directly stated in the context. "
            "Provide a concise answer."
        )
        full_prompt = f"{preamble}\n\nQuery: {prompt}"

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            response = generate_response(model, tokenizer, full_prompt, generation_config)
            response_placeholder.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


# def main():
#     st.title("Lung Cancer AI System")


#     model_path = "/home/h392x566/llama3.2-8b-train-py"  
#     st.sidebar.title("Model Settings")
#     st.sidebar.text(f"Model: {model_path}")

#     model, tokenizer = load_model_and_tokenizer(model_path)
#     generation_config = configure_generation_settings()

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Display chat history
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # User input
#     if prompt := st.chat_input("Enter your query:"):
#         st.session_state.chat_history.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Generate response
#         with st.chat_message("assistant"):
#             response_placeholder = st.empty()
#             response_placeholder.markdown("Thinking...")
#             response = generate_response(model, tokenizer, prompt, generation_config)
#             response_placeholder.markdown(response)

#         st.session_state.chat_history.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()
