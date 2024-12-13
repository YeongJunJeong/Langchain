import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import pandas as pd

data = pd.read_csv('음식점.csv', encoding = 'cp949')

# 데이터 벡터화 함수
def prepare_vector_store(dataframe):
    documents = [
        Document(page_content=row['content'], metadata={"title": row['title']})
        for _, row in dataframe.iterrows()
    ]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # FAISS 벡터 스토어 생성
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store

vector_store = prepare_vector_store(data)
retriever = vector_store.as_retriever(search_type="similarity", search_k=5)

# LangChain LLM 및 체인 설정
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 프롬프트 템플릿 설정
prompt_template = PromptTemplate(
    input_variables=["retrieved_docs", "user_query"],
    template="""
You are an expert in recommending restaurants in Daegu, South Korea. Respond kindly to users' questions and provide appropriate recommendations.
Always answer in Korean.
Follow this format for responses:
1. Restaurant: A brief description of the restaurant
2. Restaurant: A brief description of the restaurant
It is recommended to use a search engine to create concise descriptions.
"""
)

# LLM 체인 생성
qa_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.set_page_config(page_title="대푸리카(DFRC)", page_icon="🥞")
st.title('대푸리카(DFRC)')
st.caption(':blue 대구여행 추천 Chat 🥞')

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요.", key="user_input")
if user_input:
    # 검색 결과 가져오기
    search_results = retriever.get_relevant_documents(user_input)
    retrieved_docs = "\n".join(
        [f"- {doc.metadata['title']}: {doc.page_content}" for doc in search_results]
    )
    
    # 검색 결과와 사용자 질문으로 답변 생성
    response = qa_chain.run({
        "retrieved_docs": retrieved_docs,
        "user_query": user_input
    })
    
    # 답변 출력
    st.write(response)
