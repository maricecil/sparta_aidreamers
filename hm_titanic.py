#!/usr/bin/env python
# coding: utf-8

# In[1]:


#과제: 타이타닉 생존자 예측


# In[2]:


#데이터 불러오기

#라이브러리 임포트
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')


# In[3]:


#불러온 데이터 feature 분석
titanic.head()


# In[4]:


#불러온 데이터 통계 확인
titanic.describe()
#count: 해당 열에서 값이 있는 데이터의 개수 (단, 결측치는 포함하지 않음)
#mean: 해당 열의 모든 값을 더한 후, 데이터의 개수로 나눈 평균값
#std: 표준편차: 데이터가 평균을 기준으로 얼마나 퍼져 있는 지를 나타내는 지표(값이 클수록 데이터가 평균을 기준으로 더 넓게 퍼져있음) 
#min: 해당 열 데이터 가운데 최소값
#25%: 데이터의 하위 25%에 해당하는 값. 데이터를 오름차순으로 정렬했을 때, 가장 작은 25%의 값이 이 값보다 작거나 같다는 것을 의미. 즉, 데이터의 1/4 지점
#50%: 데이터의 중긴깂. 데이터를 오름차순으로 정렬했을 때, 정확히 중앙에 위치하는 값. 이 값은 데이터의 절반이 이 값보다 작거나 같다는 것을 의미. 이는 평균과는 다르게 극단적인 값의 영향을 받지 않음.
#75%: 데이터의 하위 75%에 해당하는 값. 데이터를 오름차순으로 정렬했을 때, 가장 작은 75%의 값이 이 값보다 작거나 같다는 것을 의미. 즉, 데이터의 3/4 지점
#max: 해당 열 데이터 가운데 최대값


# In[5]:


#isnull()함수와 sum()함수를 이용해 데이터의 결측치 확인
titanic.isnull().sum()
#1. isnull() 함수를 이용하여 titanic 데이터의 결측치를 파악
#2. sum() 함수를 이용하여 파악된 결측치 개수를 모두 더함

#결측치 개수
#age: 177개
#embarked: 2개
#deck: 688개
#embark_town: 2개


# In[6]:


#feature engineering
#결측치 처리

#age(나이)의 결측치 -> 중앙값 대체
#titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

#embarked(승선항구)의 결측치 -> 최빈값 대체
#titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# fillna(): 결측값을 채워주는 함수
# median(): 데이터 중간값을 의미
# inplace=True: 데이터 변경 시, 원본 데이터 자체를 수저아는 것을 의미
#mode()[0]: 최빈값 가운데, 가장 첫 번째 값을 선택


# In[7]:


#대체 결과 -> isnull()함수와 sum()함수를 이용해 확인
titanic.isnull().sum()
#결측치 처리한 age, embared에서 결측치 x


# In[9]:


#수치형으로 인코딩
#sex(성별) - 남자 -> 0,여자 -> 1
titanic['sex'] = titanic['sex'].map({'male':0, 'female':1})

#alive(생존여부) - True -> 1, False -> 0
titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0})

#embarked(승선항구) - C -> 0, Q -> 1, S -> 2
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# In[10]:


#변환 결과 head 함수 이용해 확인
print(titanic['sex'].head())


# In[11]:


#변환 결과 head 함수 이용해 확인
print(titanic['alive'].head())


# In[12]:


#변환 결과 head 함수 이용해 확인
print(titanic['embarked'].head())


# In[13]:


#전체 데이터 확인
titanic


# In[18]:


#새로운 feature 생성

#sibsp(타이타닉호에 동승한 자매 및 배우자 수)
#parch(타이타닉호에 동승한 부모 및 자식의 수))
#를 통해 family_size(가족 크기)를 생성
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1


# In[21]:


#새로운 feature를 head 함수를 이용해 확인
titanic['family_size'].head()


# In[27]:


#모델 학습시키기

#모델 학습 준비
#'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 사용.
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]


# In[29]:


#feature과 target 분리

#feature
#전체 데이터 가운데, 예측해야하는 data인 'survived' 데이터를 drop
#단, axis=1을 이용하여 survived열을 삭제
X = titanic.drop('survived', axis=1)

#target
y = titanic['survived']


# In[30]:


#Logistic Regression, Random Forest, XGBoost를 통해서 생존자를 예측하는 모델을 학습


# In[33]:


#Logistic Regression과 Random Forest -> accuracy
#XGBoost -> mean squared error
#를 통해 test data를 예측


# In[37]:


#Logistic Regression

#라이브러리 임포트
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

#예측
y_pred = model.predict(X_test)

#평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')


# In[38]:


#Random Forest

#필요 라이브러리 임포트
from sklearn.tree import DecisionTreeClassifier

#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#예측
y_pred = model.predict(X_test)

#평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test,y_pred)}')


# In[43]:


#XGBoost

#필요 라이브러리 임포트
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#데이터를 학습용/테스트용으로 분할
#X: 입력 데이터(특성, features)
#y: 목표 데이터(타겟, target)
#test_size=0.2: 전체 데이터 중 학습 데이터 80%, 테스트 데이터 20%로 분할
#random_state=42: 랜덤 시드값을 고정하여 분할율 제현을 가능하게함.

#데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#학습 데이터를 기준으로 평균과 표준편차를 계산하고, 그 값을 바탕으로 데이터를 변환
X_test_scaled = scaler.transform(X_test)
#학습 데이터에서 계산된 평균과 표준편차를 사용해 테스트 데이터를 동일한 방식으로 변환

#XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#n_estimators=100: 생성할 결정 트리의 개수를 지정. 
#트리의 개수가 많을수록 모델이 복잡해지고 학습 시간이 늘어남,
# -> 성능에 맞게 최적화해야함.

#learning_rate=0.1: 학습 속도를 조정하는 하이퍼파라미터.
#-> 학습 속도가 너무 높으면 학습이 불안정해질 수 있고, 너무 낮으면 학습 속도가 느려짐

#max_depth=3: 개별 결정 트리의 최대 깊이(노드의 깊이)를 제한.
#-> 이 값이 크면 모델이 더 복잡해지며, 과적합 위험이 있음


#모델 학습
xgb_model.fit(X_train_scaled, y_train)

#예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

#평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')
#mean_squared_error(y_test, y_pred_xgb): 실제 값(y_test)과 예측 값(y_pred_xgb) 사이의 평균 제곱 오차(MSE)를 계산


# In[ ]:




