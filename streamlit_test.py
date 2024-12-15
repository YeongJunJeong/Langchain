from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import os
import re

df = pd.read_csv(r"음식점.csv", encoding="cp949")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([("system" ,'''Your name is “가볼까?” He is a Daegu travel recommendation expert 
                                            who recommends Daegu travel destinations. You must always answer in Korean.
                                            You must speak kindly.
                                            1. Destination
                                            2. Destination
                                            All you have to do is provide 5 destinations in this format.'''),
                                           ("user", "{message}")])

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

# 대화 이력 초기화
chat_history = []

#추천 모델

#불용어 처리
korean_stop_words = [
    "이", "그", "저", "에", "가", "을", "를", "의", "은", "는", "들", "를", "과", "와", "에게", "게",
    "합니다", "하는", "있습니다", "합니다", "많은", "많이", "많은", "많이", "모든", "모두", "한", "그리고", "그런데",
    "나", "너", "우리", "저희", "이런", "그런", "저런", "어떤", "어느", "그럴", "것", "그것", "이것", "저것", 
    "그러나", "그리하여", "그러므로", "그래서", "하지만", "그럼에도", "이에", "때문에", "그래서", "그러니까", 
    "이렇게", "그렇게", "저렇게", "어떻게", "왜", "무엇", "어디", "언제", "어떻게", "어느", "모두", "모든", 
    "그래도", "하지만", "그러면", "그런데", "하지만", "이러한", "그러한", "저러한", "이러한", "이렇게", "그렇게",
    "저렇게", "어떻게", "왜", "어디", "언제", "어떻게", "모두", "모든", "몇", "누구", "무슨", "어느", "얼마나",
    "무엇", "무슨", "아무", "여기", "저기", "거기", "그곳", "이곳", "저곳", "무엇", "아무", "모두", "마치",
    "보다", "보이다", "등", "등등", "등등등"
    ]


def recommend(df, user_input, korean_stop_words):
    
    user_input_list = [user_input]
    
    all_about_data = df['all_about'].tolist()

    tfidf = TfidfVectorizer(stop_words=korean_stop_words)
    tfidf_matrix_all_about = tfidf.fit_transform(all_about_data)
    tfidf_matrix_input = tfidf.transform(user_input_list)

    # 코사인 유사도 검사
    cosine_sim = linear_kernel(tfidf_matrix_input, tfidf_matrix_all_about)

    top_place = cosine_sim.argsort()[0][-5:][::-1]

    recommended_places = []
    for idx in top_place:
        place_info = df.iloc[idx]
        recommended_places.append(f"{place_info['name']}: {place_info['info']}")

    for place in recommended_places:
        print(place)


# Streamlit 설정
st.set_page_config(
    page_title="대푸리카(DFRC)", 
    page_icon="🥞"
)

st.title('대푸리카(DFRC)')
st.caption(':blue[대구여행 추천 Chat 🥞]')

# 초기 메시지 히스토리
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

messages = st.container()
user_input = st.chat_input("질문을 입력하세요.", key="user_input")

if user_input:
    # 사용자 입력 기록 추가
    st.session_state['chat_history'].append({"role": "user", "message": user_input})

    # AI 응답 생성
    ai_response, st.session_state['chat_history'] = response(user_input, st.session_state['chat_history'])

    # 패턴 매칭 및 출력
    pattern2 = re.findall(r'^.*?(?=1\.)', ai_response, re.DOTALL)

    if pattern2:
        for item in pattern2:
            with messages:
                st.write(f"AI: {item.strip()}")
                # 추천 함수 호출
                recommend(None, user_input, None)  # df와 korean_stop_words는 실제 값으로 대체하세요.
    else:
        with messages:
            st.write(f"AI: {ai_response}")

# 채팅 기록 출력
with messages:
    for msg in st.session_state['chat_history']:
        role = "사용자" if msg['role'] == "user" else "AI"
        st.write(f"{role}: {msg['message']}")
