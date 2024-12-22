from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import pandas as pd
import streamlit as st

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    
#if "openai_model" not in st.session_state:
    #st.session_state["openai_model"] = "gpt-3.5-turbo"

# CSV 파일 경로
csv_file_path = '음식점.csv'  # CSV 파일 경로를 여기에 입력하세요

# 데이터 로드 및 준비
data = pd.read_csv(csv_file_path, encoding = 'cp949')

# 필요한 칼럼이 있는지 확인
if 'all_about' not in data.columns or '상호명' not in data.columns:
    raise ValueError("CSV 파일에 'all about' 또는 'name' 칼럼이 없습니다.")

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings()

# "all about" 및 "name" 텍스트를 벡터화
vectorstore = FAISS.from_texts(data['all_about'].tolist(), embeddings)

# Streamlit UI 설정
st.title("🍔 대푸리카 (DFRC)")
st.caption("🍜 대구광역시 맛집 추천 챗봇 서비스")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "예) 수성못 근처에 가족들과 조용하게 외식할 곳을 추천해줘"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 입력 텍스트 설정
    input_text = prompt

    # 입력 텍스트와 가장 유사한 5개 항목 검색
    similarities = vectorstore.similarity_search(input_text, k=5)

    # 유사한 항목들의 name 추출
    similar_names = [data.loc[data['all_about'] == match.page_content, '상호명'].values[0] for match in similarities]

    # ChatGPT 초기화
    llm = OpenAI(temperature=0.7)

    # ChatGPT로 간단한 설명 생성
    explanations = []
    for name in similar_names:
        response = llm(f"'{name}'은 대구광역시에 있는 식당입니다. 관련된 내용을 한 줄로 매우 간단히 설명해주세요. 항상 친절하게 존댓말로 해주세요")
        explanations.append(response)

    # 결과 생성 및 출력
    result = "\n\n".join([f"{idx}. {name}:\n{explanation}" for idx, (name, explanation) in enumerate(zip(similar_names, explanations), start=1)])

    # 답변 추가
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
