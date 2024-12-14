import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# OpenAI API 키 설정 및 초기화
llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are an expert in recommending great restaurants and delicious cafes in Daegu, South Korea.  
Listen carefully to the questions and recommend places relevant to the query.  
Always respond with recommendations when asked.  
Be polite and explain in Korean.  
Provide 5 concise examples with a brief description for each.'''),
    ("user", "{message}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# 사용자 입력과 채팅 기록을 관리하는 함수
def response(message, history):
    history_langchain_format = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_langchain_format.append(msg)
        elif isinstance(msg, AIMessage):
            history_langchain_format.append(msg)

    # 새로운 사용자 메시지 추가
    history_langchain_format.append(HumanMessage(content=message))

    # LangChain ChatOpenAI 모델을 사용하여 응답 생성
    gpt_response = chain.invoke({"message" : message})

    # 생성된 AI 메시지를 대화 이력에 추가
    history_langchain_format.append(AIMessage(content=gpt_response))

    return gpt_response, history_langchain_format

# 챗봇 UI 구성
st.set_page_config(
    page_title="대푸리카(DFRC)", 
    page_icon="🥞")

st.title('대푸리카(DFRC)')
st.caption(':blue 대구여행 추천 Chat 🥞')
user_input = st.chat_input("질문을 입력하세요.", key="user_input")
messages = st.container(height=400)

# 대화 이력 저장을 위한 세션 상태 사용
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if user_input:
    ai_response, new_history = response(user_input, st.session_state['chat_history'])
    st.session_state['chat_history'] = new_history

    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            messages.chat_message("user").write(message.content)
        if isinstance(message, AIMessage):
            messages.chat_message("assistant").write(message.content)
