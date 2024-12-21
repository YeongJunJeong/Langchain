from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import pandas as pd
import streamlit as st
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Streamlit 상태 초기화
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# CSV 파일 경로 설정
csv_file_path = '음식점.csv'  # CSV 파일 경로를 입력하세요

# 데이터 로드 및 준비
try:
    data = pd.read_csv(csv_file_path, encoding='cp949')
except Exception as e:
    st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
    st.stop()

# 필요한 칼럼 확인
if 'all_about' not in data.columns or '상호명' not in data.columns:
    st.error("CSV 파일에 'all_about' 또는 '상호명' 칼럼이 없습니다.")
    st.stop()

# 텍스트 길이 제한 (필요 시 조정)
MAX_TEXT_LENGTH = 1000
data['all_about'] = data['all_about'].apply(lambda x: x[:MAX_TEXT_LENGTH] if len(x) > MAX_TEXT_LENGTH else x)

# 임베딩 초기화
embeddings = OpenAIEmbeddings()

# FAISS Vectorstore 생성 함수 (배치 처리 및 오류 핸들링 포함)
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def create_faiss_vectorstore(texts, embeddings):
    all_vectors = []
    batch_size = 10  # 배치 크기 설정

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_vectors = embeddings.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)
            time.sleep(1)  # API 호출 간 대기 시간 추가
        except Exception as e:
            st.warning(f"임베딩 생성 중 오류 발생 (배치 {i // batch_size + 1}): {str(e)}")
            raise e

    return FAISS.from_embeddings(all_vectors, embeddings)

# 벡터스토어 생성
try:
    vectorstore = create_faiss_vectorstore(data['all_about'].tolist(), embeddings)
except Exception as e:
    st.error(f"벡터스토어 생성 중 오류 발생: {str(e)}")
    st.stop()

# Streamlit UI 설정
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 입력 텍스트 설정
    input_text = prompt

    # 입력 텍스트와 가장 유사한 5개 항목 검색
    try:
        similarities = vectorstore.similarity_search(input_text, k=5)
        similar_names = [data.loc[data['all_about'] == match.page_content, '상호명'].values[0] for match in similarities]
    except Exception as e:
        st.error(f"유사 항목 검색 중 오류 발생: {str(e)}")
        st.stop()

    # ChatGPT 초기화
    llm = OpenAI(temperature=0.7)

    # ChatGPT로 간단한 설명 생성
    explanations = []
    for name in similar_names:
        try:
            response = llm(
                f"""'{name}'와 관련된 내용을 간단히 설명해주세요. 당신은 대구광역시 맛집추천 전문가입니다."""
            )
            explanations.append(response)
        except Exception as e:
            explanations.append(f"'{name}'에 대한 설명 생성 중 오류 발생: {str(e)}")

    # 결과 생성 및 출력
    result = "\n".join([f"{idx}. {name} : {explanation}" for idx, (name, explanation) in enumerate(zip(similar_names, explanations), start=1)])

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
