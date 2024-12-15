import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import re

# 데이터 로드
df = pd.read_csv("음식점.csv", encoding="cp949")

# LangChain 설정
llm = ChatOpenAI()

system_message = SystemMessagePromptTemplate.from_template(''' You are a recommendation expert who recommends restaurants in Daegu. 
                                            You must always answer in Korean.
                                            You must speak kindly.
                                            1. restaurant
                                            2. restaurant
                                            All you have to do is provide 5 restaurants in this format.''')
human_message = HumanMessagePromptTemplate.from_template("{message}")
prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# 불용어 처리 함수
korean_stop_words = ["이", "그", "저", "에", "가", "을", "를", "의", "은", "는", "들", "를", "과", "와", "에게", "게",
    "합니다", "하는", "있습니다", "합니다", "많은", "많이", "많은", "많이", "모든", "모두", "한", "그리고", "그런데",
    "나", "너", "우리", "저희", "이런", "그런", "저런", "어떤", "어느", "그럴", "것", "그것", "이것", "저것", 
    "그러나", "그리하여", "그러므로", "그래서", "하지만", "그럼에도", "이에", "때문에", "그래서", "그러니까", 
    "이렇게", "그렇게", "저렇게", "어떻게", "왜", "무엇", "어디", "언제", "어떻게", "어느", "모두", "모든", 
    "그래도", "하지만", "그러면", "그런데", "하지만", "이러한", "그러한", "저러한", "이러한", "이렇게", "그렇게",
    "저렇게", "어떻게", "왜", "어디", "언제", "어떻게", "모두", "모든", "몇", "누구", "무슨", "어느", "얼마나",
    "무엇", "무슨", "아무", "여기", "저기", "거기", "그곳", "이곳", "저곳", "무엇", "아무", "모두", "마치",
    "보다", "보이다", "등", "등등", "등등등"]

def preprocess_text(text, stop_words):
    for word in stop_words:
        text = text.replace(word, "")
    return text

df['all_about'] = df['all_about'].apply(lambda x: preprocess_text(x, korean_stop_words))

# 추천 함수
def recommend(df, user_input, stop_words):
    user_input_list = [user_input]
    all_about_data = df['all_about'].tolist()
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix_all_about = tfidf.fit_transform(all_about_data)
    tfidf_matrix_input = tfidf.transform(user_input_list)
    cosine_sim = linear_kernel(tfidf_matrix_input, tfidf_matrix_all_about)
    top_place = cosine_sim.argsort()[0][-5:][::-1]

    recommended_places = []
    for idx in top_place:
        place_info = df.iloc[idx]
        recommended_places.append(f"{place_info['name']}: {place_info['info']}")
    return recommended_places

# Streamlit UI 설정
st.title('대푸리카 (DFRC)')
st.caption(':blue[대구여행 추천 Chat] 🥞')

# 대화 이력 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.chat_input("질문을 입력하세요.")
if user_input:
    gpt_response, _ = response(user_input, st.session_state["chat_history"])
    st.session_state["chat_history"].append(f"AI: {gpt_response}")

    # 추천 실행
    recommendations = recommend(df, user_input, korean_stop_words)
    for rec in recommendations:
        st.write(rec)
