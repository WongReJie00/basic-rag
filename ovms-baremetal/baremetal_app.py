


import streamlit as st
import requests

st.title("🦙 Simple OVMS Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Waiting for LLM response..."):
            try:
                data = {
                    "model": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.1
                }
                response = requests.post(
                    "http://localhost:8005/v3/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                else:
                    st.error(f"LLM error: {response.status_code} {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
