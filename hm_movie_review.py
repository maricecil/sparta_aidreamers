#!/usr/bin/env python
# coding: utf-8

# # 영화 감성 리뷰
# ### Netflix의 영화 리뷰 데이터를 사용하여, 리뷰의 평점을 예측해보고, 긍정과 부정 감정을 분류

# #### 데이터셋 불러오기

# In[29]:


#필요 라이브러리 임포트
import pandas as pd

#데이터 불러오기
df = pd.read_csv('netflix_reviews.csv')


# In[30]:


#데이터 출력(상단5개)
df.head()


# In[31]:


#데이터 출력(하단5개)
df.tail()


# In[32]:


#shape 출력

# 데이터프레임의 모양 출력
print(f"Shape of the dataset: {df.shape}")

# 데이터프레임의 열 이름 출력
print(f"Columns in the dataset: {df.columns}")


# ---

# #### 데이터 전처리

# In[33]:


#데이터 전처리
def preprocess_text(text): #텍스트 데이터를 입력받아 전처리된 결과를 반환하는 함수. 입력 변수는 text.
    if isinstance(text, float): #text라는 변수에 담긴 값이 실수(float) 타입인지 검사
        return "" #만약 text가 실수인 경우, 이후의 문자열 처리 과정에서 오류가 발생할 수 있기 때문에, 이와 같은 실수 타입의 데이터는 빈 문자열 ""로 변환
    text = text.lower() #대문자를 소문자로 변환
    text = re.sub(r'[^\w\s]', '', text) #정규 표현식을 사용하여 구두점(마침표, 쉼표 등)을 제거. 단어와 공백(\w와\s)을 제외한 모든 문자를 제거하는 역할
    text = re.sub(r'\d+', '', text) #숫자를 모두 제거, 정규 표현식 \d+는 하나 이상의 숫자를 의미하며, 이를 빈 문자열로 대체
    text = text.strip() #text.strip()은 문자열 양 끝에 있는 불필요한 공백을 제거합니다. 텍스트 중간에 있는 공백은 그대로 두지만, 시작이나 끝부분의 공백은 제거
    return text


# In[34]:


#feature 분석(EDA)

# 라이브러리 임포트
import seaborn as sns
import matplotlib.pyplot as plt

# 리뷰 점수에 대한 빈도를 계산
score_counts = df['score'].value_counts().sort_index()
#value_counts(): df['score']에 있는 각 점수의 빈도를 계산. 이 함수는 각 점수가 몇 번 나타났는지 계산하여 반환
#sort_index(): 점수의 크기 순으로 정렬 (예. 1점, 2점, 3점 ...)

# 바 그래프 생성
sns.barplot(x=score_counts.index, y=score_counts.values, palette="Greens")
# x=score_counts.index: x축에 각 점수(1,2,3,4,5)를 배치
# y=score_counts.value: y축에 각 점수에 해당하는 리뷰의 빈도를 배치
# palette: 막대그래프 색상 변경

plt.xlabel('Score')  # x축 라벨
plt.ylabel('Count')  # y축 라벨
plt.title('Distribution of Scores')  # 그래프 제목
plt.show() #그래프 출력


# ---

# ### 리뷰 예측 모델 학습시키기(LSTM)

# - 로지스틱 회귀 등을 사용하여, 리뷰에 대한 점수를 예측

# In[36]:


#라이브러리 임포트
import pandas as pd  # 데이터를 다루기 위한 라이브러리. 데이터프레임 형식으로 데이터 처리 가능.
import torch  # 딥러닝 프레임워크로, 텐서 연산 및 신경망 모델 구축을 지원.
import torch.nn as nn  # PyTorch의 신경망 모듈로, 다양한 레이어와 손실 함수를 제공.
import torch.optim as optim  # PyTorch의 최적화 모듈로, 모델의 가중치 업데이트를 위한 다양한 옵티마이저 제공.
from torchtext.data.utils import get_tokenizer  # 텍스트 데이터를 토큰화하는 함수로, 텍스트 전처리 시 사용.
from torchtext.vocab import build_vocab_from_iterator  # 토큰화된 텍스트로부터 어휘 사전(vocab)을 생성하는 함수.
from torch.utils.data import DataLoader, Dataset  # PyTorch의 데이터 로딩을 위한 클래스와 배치 단위로 데이터를 처리하는 도구.
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 테스트용으로 나누는 함수.
from sklearn.preprocessing import LabelEncoder  # 범주형 데이터를 숫자로 인코딩하는 도구.
import numpy as np  # 수치 연산을 위한 라이브러리로, 다차원 배열과 수학적 연산을 처리.


# In[ ]:


#

