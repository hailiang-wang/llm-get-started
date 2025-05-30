#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2025 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /c/Users/Administrator/courses/LLms/llm-get-started/004_chatbot_langchain/app.py
# Author: Hai Liang Wang
# Date: 2025-05-30:18:19:48
#
#===============================================================================

"""
Building a Local AI Chatbot Using Streamlit, LangChain, and Ollama
https://medium.com/@Shamimw/building-a-local-ai-chatbot-using-streamlit-langchain-and-ollama-484b82083988
"""
__copyright__ = "Copyright (c) 2025 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-05-30:18:19:48"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

# Get ENV
import env3
ENV = env3.read_env()

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ---- Streamlit Setup ---- #
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

# ---- Sidebar Inputs ---- #
st.sidebar.header("Settings")

# Dropdown for model selection
model_options = ["deepseek-r1:14b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0)

# Inputs for max history and context size
MAX_HISTORY = st.sidebar.number_input("Max History", min_value=1, max_value=100, value=2, step=1)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", min_value=1024, max_value=16384, value=8192, step=1024)

# ---- Function to Clear Memory When Settings Change ---- #
def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)  # Reset memory

# Clear memory if settings are changed
if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
    clear_memory()
    st.session_state.prev_context_size = CONTEXT_SIZE

# ---- Initialize Chat Memory ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ---- LangChain LLM Setup ---- #
llm = ChatOllama(model=MODEL, streaming=True)

# ---- Prompt Template ---- #
prompt_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="{history}\nUser: {human_input}\nAssistant:"
)

chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.memory)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Trim Function (Removes Oldest Messages) ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:  # Each cycle has 2 messages (User + AI)
        st.session_state.chat_history.pop(0)  # Remove oldest User message
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)  # Remove oldest AI response

# ---- Handle User Input ---- #
if prompt := st.chat_input("Say something"):
    # Show User Input Immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})  # Store user input

    # Trim chat history before generating response
    trim_memory()

    # ---- Get AI Response (Streaming) ---- #
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in chain.stream({"human_input": prompt}):
            if isinstance(chunk, dict) and "text" in chunk:
                text_chunk = chunk["text"]
                full_response += text_chunk
                response_container.markdown(full_response)

    # Store response in session_state
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Trim history after storing the response
    trim_memory()