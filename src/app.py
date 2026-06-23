import streamlit as st

# Configure the presentation layout
st.set_page_config(page_title="RAG Complaint Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Enterprise RAG Complaint Management Chatbot")
st.subheader("10 Academy Week 7 Capstone Pipeline")
st.markdown("---")

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    chunk_size = st.slider("Chunk Size (Words)", 200, 1000, 500, step=50)
    overlap = st.slider("Overlap Size", 0, 200, 50, step=10)
    st.success("Virtual Environment Connected!")

# Main chat interface state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your RAG assistant. Ask me anything about the uploaded company data or user complaints."}
    ]

# Render persistent historical context window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User prompt logic
if user_query := st.chat_input("Ask a question about user complaints..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # RAG Generation Placeholder
    with st.chat_message("assistant"):
        with st.spinner("Searching vector knowledge base and synthesizing answer..."):
            # We will plug in our real retriever function right here!
            mock_response = f"Received: '{user_query}'. (Retrieval engine is initializing... ready to index with chunk size {chunk_size}!)"
            st.write(mock_response)
    st.session_state.messages.append({"role": "assistant", "content": mock_response})