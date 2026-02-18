# streamlit + langchain + ollama (LLM-gemma2:2b model)
# import required libraries

import os
import streamlit as st

# import ollama properly
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# step 1 - create prompt template
# this defines how AI should behave and how it receives user input

prompt = ChatPromptTemplate.from_messages(
    [
        # system message defines AI behaviour
        ("system", "you are a helpful assistant. please respond clearly to the question asked"),
        
        # human message contains placeholder {question}
        ("human", "Question : {question}")
    ]
)

# step 2 - streamlit app UI

# app title
st.title("langchain demo with gemma model (ollama)")

# text input box for users question
input_txt = st.text_input("what question do you have in your mind?")

# step 3 - load ollama model

# load local gemma model
LLM = ChatOllama(model="gemma2:2b")

# convert output model to string
output_parser = StrOutputParser()

# create langchain pipeline (prompt --> model --> output_parser)
chain = prompt | LLM | output_parser

# step 4 - run the model when user inputs the question
if input_txt:
    response = chain.invoke({"question": input_txt})
    st.write(response)
