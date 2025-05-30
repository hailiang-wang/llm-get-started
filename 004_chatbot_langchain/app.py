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

from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Get ENV
import env3
ENV = env3.read_env()

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

class State(TypedDict):
    '''
    Note that we have added a new language input to the prompt. Our application now has two parameters-- the input messages and language. We should update our application's state to reflect this.
    https://python.langchain.com/docs/tutorials/chatbot/
    '''
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


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

# Clear memory if settings are changed
if "prev_context_size" not in st.session_state or st.session_state.prev_context_size != CONTEXT_SIZE:
    clear_memory()
    st.session_state.prev_context_size = CONTEXT_SIZE

# ---- Initialize Chat Memory ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- LangChain LLM Setup ---- #
# https://python.langchain.com/docs/tutorials/chatbot/
llm = ChatOllama(model=MODEL, streaming=True)
messages = []

# Define a new graph
workflow = StateGraph(state_schema=State)

# ---- Prompt Template ---- #
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """1. You are a helpful assistant. Answer all questions to the best of your ability in {language}.\n
            2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
            3. Keep the answer crisp and limited to 3,4 sentences. \n""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Managing Conversation History
trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# ---- Define the function that calls the model ---- #
def call_model(state: State):
    state["messages"] = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)

    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

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

    # Understand memory object
    # https://github.com/hailiang-wang/llm-get-started/issues/3#issuecomment-2922227938
    # print("what is memory")
    # print(st.session_state.memory)

    # ---- Get AI Response (Streaming) ---- #
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        config = {"configurable": {"thread_id": "abc123"}}
        language = "Chinese"

        input_messages = messages + [HumanMessage(prompt)]
        # if getting reply directly, without streaming
        # output = app.invoke({"messages": input_messages}, config)
        # content='<think>\n\n</think>\n\nHi Bob! Welcome. How can I assist you today? 😊' additional_kwargs={} response_metadata={'model': 'deepseek-r1:14b', 'created_at': '2025-05-30T12:34:44.6516712Z', 'done': True, 'done_reason': 'stop', 'total_duration': 10507775500, 'load_duration': 4320399000, 'prompt_eval_count': 8, 'prompt_eval_duration': 1645474100, 'eval_count': 19, 'eval_duration': 4541116600, 'message': Message(role='assistant', content='', images=None, tool_calls=None)} id='run-3cfa31e0-9663-4dc5-a0df-2d255a5bfdca-0' usage_metadata={'input_tokens': 8, 'output_tokens': 19, 'total_tokens': 27}

        for chunk, metadata in app.stream(
                {"messages": input_messages, "language": language},
                config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):  # Filter to just model responses
                    text_chunk = chunk.content
                    full_response += text_chunk
                    response_container.markdown(full_response)


    # Store response in session_state
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Trim history after storing the response
    trim_memory()