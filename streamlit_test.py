import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # 대화 내역 저장

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"  # 기본 모델 설정

# OpenAI LLM 초기화
llm = ChatOpenAI()

# 프롬프트 템플릿 설정
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

# 대화 응답 생성 함수
def generate_response(user_message):
    # LangChain을 통해 AI 응답 생성
    gpt_response = chain.invoke({"message": user_message})
    return gpt_response

# Streamlit UI 설정
st.set_page_config(
    page_title="대푸리카(DFRC)", 
    page_icon="🥞"
)

st.title('대푸리카(DFRC)')
st.caption(':blue[대구여행 추천 Chat 🥞]')

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요.", key="user_input")

# 사용자 메시지 입력 시 처리
if user_input:
    # 사용자 메시지를 기록
    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    # 화면에 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # AI 응답 생성 및 기록
    ai_response = generate_response(user_input)
    st.session_state["chat_history"].append(AIMessage(content=ai_response))
    # 화면에 AI 응답 출력
    st.chat_message("assistant").write(ai_response)

# 기존 대화 이력을 순서대로 출력
if st.session_state["chat_history"]:
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)
