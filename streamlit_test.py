import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 데이터 로드
df = pd.read_csv('음식점.csv', encoding='cp949')

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

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
    gpt_response = chain.invoke({"message": message})

    # 생성된 AI 메시지를 대화 이력에 추가
    history_langchain_format.append(AIMessage(content=gpt_response))

    return gpt_response, history_langchain_format

chat_history = []

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

    # 모든 열 데이터를 결합
    df['combined'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    combined_data = df['combined'].tolist()

    # TF-IDF 처리
    tfidf = TfidfVectorizer(stop_words=korean_stop_words)
    tfidf_matrix_combined = tfidf.fit_transform(combined_data)
    tfidf_matrix_input = tfidf.transform(user_input_list)

    # 코사인 유사도 계산
    cosine_sim = linear_kernel(tfidf_matrix_input, tfidf_matrix_combined)

    # 가장 유사도가 높은 5개 추천
    top_place_indices = cosine_sim.argsort()[0][-5:][::-1]

    # 추천 장소 리스트 생성
    recommended_places = []
    for idx in top_place_indices:
        recommended_places.append(df.iloc[idx].to_dict())  # 행 데이터를 딕셔너리로 변환

    return recommended_places

# GPT로 설명 생성
def generate_place_descriptions(places):
    # 장소 정보를 문자열로 정리
    place_details = "\n\n".join(
        [f"장소 {i+1}:\n이름: {place['name']}\n설명: {place['info']}\n주소: {place['address']}\n연락처: {place['phone']}"
         for i, place in enumerate(places)]
    )

    # GPT에 전달하여 설명 생성
    gpt_response = chain.invoke({"place_details": place_details})
    return gpt_response

# 챗봇 UI 구성
st.set_page_config(
    page_title="대푸리카(DFRC)", 
    page_icon="🥞"
)

st.title('대푸리카(DFRC)')
st.caption(':blue 대구여행 추천 Chat 🥞')
user_input = st.chat_input("질문을 입력하세요.", key="user_input")
messages = st.container()

# 대화 이력 저장을 위한 세션 상태 사용
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if user_input:
    # 추천 결과 생성
    recommended_places = recommend(df, user_input, korean_stop_words)

    # GPT 설명 생성
    gpt_explanation = generate_place_descriptions(recommended_places)

    # 대화 메시지 출력
    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            messages.chat_message("user").write(message.content)
        if isinstance(message, AIMessage):
            messages.chat_message("assistant").write(message.content)

    # 추천 결과 및 GPT 설명 출력
    with st.container():
        st.subheader("GPT가 추천한 장소 설명:")
        st.write(gpt_explanation)

        st.subheader("추천된 장소 상세 정보:")
        for place in recommended_places:
            st.write(place)
