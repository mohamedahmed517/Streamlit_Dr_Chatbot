import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

user = "mohamed517"
hf_repo_name = "Dr_Chatbot"

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(f"{user}/{hf_repo_name}").to(device)
tokenizer = T5Tokenizer.from_pretrained(f"{user}/{hf_repo_name}")

model.eval()

# Response generation function
def generate_response_top_k_top_p(question, model, tokenizer, max_length=64, top_k=50, top_p=0.95, temperature=1.0):
    formatted_question = f"Answer the following question: {question}"

    inputs = tokenizer(
        formatted_question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)  # <-- Ensure inputs are on the same device

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Dr Chatbot")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How Can I Help You?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bot response generation
def generation(question):
    model_answer = generate_response_top_k_top_p(question, model, tokenizer)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": model_answer})
    return model_answer

# User input section
question = st.chat_input("How Can I Help You?")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    bot_response = generation(question)

    with st.chat_message("assistant"):
        st.markdown(bot_response)
