import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Streamlit page configuration
st.set_page_config(page_title="AI System Interactive Chat", layout="wide")

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

    # Add pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # model.to("cuda")  # Ensure model is fully on CUDA
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

@torch.inference_mode()
def generate_response(model, tokenizer, prompt, generation_config):
    # Tokenize the input with padding and return tensors
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  # Adjust max_length based on your use case
    ).to("cuda")

    # Use pad_token_id explicitly
    pad_token_id = tokenizer.pad_token_id
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=generation_config.max_new_tokens,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        do_sample=generation_config.do_sample,
        pad_token_id=pad_token_id  # Set pad_token_id to avoid warnings
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# Sidebar configuration for generation settings
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

# Main function to run the app
def main():
    st.title("AI System Interactive Chat")

    # Path to the locally stored model
    model_path = "/home/xuan/llama3.2-8b-train-py"  # Path to your local model
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

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            response = generate_response(model, tokenizer, prompt, generation_config)
            response_placeholder.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
