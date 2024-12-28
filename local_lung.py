import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load the local Llama model for text generation
llama_model_path = "/home/xuan/llama3.2-8b-train-py"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

# Load the local model for embeddings (if required)
embedding_model_path = "/home/xuan/llama3.2-8b-train-py"
embedder = SentenceTransformer(embedding_model_path)

# Function to generate a response
def generate_response(prompt, max_tokens=512, temperature=0.7):
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(
        inputs["input_ids"],
        max_length=max_tokens,
        temperature=temperature,
        pad_token_id=llama_tokenizer.eos_token_id,
    )
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to get embeddings
def get_embedding(text):
    return embedder.encode(text)

# Streamlit app setup
st.title("Local ChatGPT using Llama 3.1 with Personal Memory 🧠")
st.caption("Each user gets their own personalized memory space!")

# Initialize session state for chat history and previous user ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_user_id" not in st.session_state:
    st.session_state.previous_user_id = None

# Sidebar for user authentication
with st.sidebar:
    st.title("User Settings")
    user_id = st.text_input("Enter your Username", key="user_id")

    # Check if user ID has changed
    if user_id != st.session_state.previous_user_id:
        st.session_state.messages = []  # Clear chat history
        st.session_state.previous_user_id = user_id  # Update previous user ID

    if user_id:
        st.success(f"Logged in as: {user_id}")

# Main chat interface
if user_id:  # Only show chat interface if user is "logged in"
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What is your message?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Generate response using the local Llama model
                full_response = generate_response(prompt)
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "I apologize, but I encountered an error generating the response."
                message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("👍 Please enter your username in the sidebar to start chatting!")
