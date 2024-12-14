import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Step 1: 데이터 로드 및 임베딩 생성
@st.cache_resource
def load_and_index_data(csv_path, model_name="all-MiniLM-L6-v2"):
    # CSV 데이터 로드
    data = pd.read_csv(csv_path, encoding = "cp949")
    model = SentenceTransformer(model_name)
    
    # '상호명', '업종분류명', '도로명주소', '리뷰'를 하나로 합쳐 검색 데이터 생성
    data['combined'] = data['상호명'] + " " + data['업종분류명'] + " " + data['도로명주소'] + " " + data['리뷰']
    
    # 데이터 임베딩
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=False)
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return data, model, index

# Step 2: 음식점 검색
def search_restaurant(query, data, model, index, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]]
    return results

# Step 3: 대화 흐름 설정
def setup_chain(data, model, index):
    # LangChain 메모리
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # LangChain 프롬프트 설정
    prompt = ChatPromptTemplate.from_template(
        template="사용자의 질문에 적합한 음식점을 추천하세요.\n질문: {query}\n답변:"
    )
    
    # Conversational Chain 설정
    chain = ConversationalRetrievalChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=lambda q: search_restaurant(q, data, model, index),
        memory=memory,
        prompt=prompt,
    )
    return chain

# Step 4: Streamlit UI
def main():
    st.set_page_config(page_title="음식점 추천", layout="wide")
    st.title("🍴 음식점 추천 시스템")
    
    # 데이터 로드 및 인덱스 생성
    data, model, index = load_and_index_data("음식점.csv")
    
    # LangChain Chain 설정
    chain = setup_chain(data, model, index)
    
    # 사용자 입력
    query = st.text_input("질문을 입력하세요", placeholder="예: 강남에 있는 맛있는 이탈리안 레스토랑 추천해줘")
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # 대화 처리
    if query:
        results = search_restaurant(query, data, model, index)
        context = "\n".join(results['combined'].tolist())
        response = chain.run(query=query)
        
        # 대화 이력 저장
        st.session_state["chat_history"].append({"query": query, "response": response})
        
        # 결과 표시
        st.subheader("추천 결과")
        for _, row in results.iterrows():
            st.write(f"**상호명**: {row['상호명']}")
            st.write(f"**업종분류명**: {row['업종분류명']}")
            st.write(f"**도로명주소**: {row['도로명주소']}")
            st.write(f"**리뷰**: {row['리뷰']}")
            st.markdown("---")
        
        st.subheader("AI 답변")
        st.write(response)
    
    # 대화 이력 표시
    st.subheader("대화 이력")
    for history in st.session_state["chat_history"]:
        st.write(f"**질문**: {history['query']}")
        st.write(f"**답변**: {history['response']}")
        st.markdown("---")

if __name__ == "__main__":
    main()
