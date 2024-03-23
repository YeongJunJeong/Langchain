from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#OpenAi API Key
llm = ChatOpenAI(openai_api_key = "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")

#Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a Daegu travel expert who recommends 
     Daegu tourist attractions to people. 
     You must always answer in Korean, and you must always recommend
     3 suitable travel destinations, 3 suitable restaurants, 
     and 3 suitable accommodations with brief explanations.
     Typing “고마워” will say thank you and goodbye.
     '''),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

user_input = input("사용자: ")
result = chain.invoke({"input" : user_input})
print(result)







#import pandas as pd
#import folium
#import re
#import streamlit as st
#from streamlit_folium import folium_static

# while True:
#     user_input = input("사용자: ")
#     result = chain.invoke({"input" : user_input})
#     print(result)

#     if user_input.strip() == "고마워":
#         print("언제든지 또 찾아주세요 :)")
#         break
        

# #Streamlit Display
# st.balloons()
# st.title('가볼까?')
# st.caption('대구여행은 _:blue[가볼까?]_ 와 함께 :sunglasses:')
# user_input = st.text_input('어떤 여행을 떠나시나요?')
# st.caption('예) 친구들과 1박 2일동안 대구를 여행할 계획이야')
# st.caption('예) 부모님과 하루 정도 대구를 여행할 계획이야. 사람들이 많이 없는 조용하고 한적한 곳으로 알려줘.')
# if st.button('여행지 추천받기'):
#     with st.spinner('골똘히 고민 중...'):
#         result = chain.invoke({"input" : user_input})
#     st.write(result)
    
#     text = result 

#     pattern = r"(.+?)\(위도: (\d+\.\d+), 경도: (\d+\.\d+)\)"
#     matches = re.findall(pattern, text)
#     coordinates = [(match[0].strip(), float(match[1]), float(match[2])) for match in matches]
    
#     # folium map
#     m = folium.Map(location = [35.864592, 128.593334], zoom_start= 14)
        
#     for coordinate in coordinates:
#         name, latitude, longitude = coordinate
#         folium.Marker(
#             location=[latitude, longitude],
#             popup=name,
#             tooltip="추천 장소"
#         ).add_to(m)

#     st_data = folium_static(m, width = 725)
