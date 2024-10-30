#!/usr/bin/env python
# coding: utf-8

# # 타이타닉 생존자 예측
# ### : 타이타닉 탑승객 데이터셋을 활용해 생존자를 예측하는 모델을 만드는 프로젝트

# ---

# #### 데이터 불러오기

# In[1]:


#라이브러리 임포트
import pandas as pd
import seaborn as sns

#데이터 로드
titanic = sns.load_dataset('titanic')


# ---

# #### feature 분석

# In[2]:


#불러온 데이터 feature 분석
titanic.head()


# In[3]:


#불러온 데이터 통계 확인
titanic.describe()
#count: 해당 열에서 값이 있는 데이터의 개수 (단, 결측치는 포함하지 않음)
#mean: 해당 열에서 모든 값을 더한 후, 데이터의 개수로 나눈 평균값
#std: 표준편차, 데이터가 평균을 기준으로 얼마나 퍼져있는 지를 나타내는 지표 (값이 클수록 데이터가 평균을 기준으로 더 넓게 퍼져있음)
#min: 해당 열 데이터 가운데 최소값
#25%: 데이터가 하위 25%에 해당하는 값. 데이터를 오름차순으로 정렬했을 때, 가장 작은 25%의 값이 이 값보다 작거나 같다는 것을 의미. 즉, 데잍의 1/4 지점.
#50%: 데이터의 중간값. 데이터를 오름차순으로 정렬했을 때, 정확히 중앙에 위치하는 값. 이 값은 데이터의 절반이 이 값보다 작거나 같다는 것을 의미. 이는 평균과는 다르게 극단적인 값의 영향을 받지 않음.
#75%: 데이터의 하위 75%에 해당하는 값. 데이터를 오름차순으로 정렬했을 때, 가장 작은 75%의 값이 이 값보다 작거나 같다는 것을 의미. 즉, 데이터의 3/4 지점.
#max: 해당 열 데이터 가운데 최대값


# In[4]:


#isnull()함수와 sum() 함수를 이용해 데이터의 결측치 확인
titanic.isnull().sum()
#1. isnull() 함수를 이용하여 titanuc 데이터의 결측치를 파악
#2. sum() 함수를 이용하여 파악된 결측치 개수를 모두 더함

#결측치 출력 리스트
#age: 177개
#embarked: 2개
#deck: 688개
#embark_town: 2개


# ---

# #### feature engineering

# In[5]:


#결측치 처리

#age(나이)의 결측치 -> 증잉깂으로 대체
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

#embarked(승선항구)의 결측치 -> 최빈값으로 대체
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0]) 
#titanic['embarked'].mode() -> 여러 최빈값을 반환.
#titanic['embarked'].mode()[0] -> 여러 최빈값 중 첫 번째 최빈값만 선택

#fillna(): 결측값을 채워주는 함수
#median(): 데이터 중간값을 의미
#inplace=True: 데이터 변경 시, 원본 데이터 자체를 수정하는 것을 의미
#mode(): 최빈값을 의미


# In[6]:


#전처리 이후 데이터 확인
#isnull().sum() 함수를 이용해 확인
titanic.isnull().sum()
#결측치 처리가 완료된 age, embarked에서는 결측치 확인 안되는 것 확인


# In[7]:


#수치형으로 인코딩

#sex(성별) - 남자:0, 여자:1
#alive(생존 여부) - yes:1, no:0
#embarked - C:0, Q:1, S:2

titanic['sex'] = titanic['sex'].map({'male':0, 'female':1})
titanic['alive'] = titanic['alive'].map({'yes':1, 'no':0})
titanic['embarked'] = titanic['embarked'].map({'C':0, 'Q':1, 'S':2})


# In[8]:


#변환 결과 확인 - head 함수 활용
titanic.head()


# In[9]:


#새로운 feature 생성

#sibsp(타이타닉호에 동승한 자매 및 배우자 수), parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성
#sibsp: 타이타닉호에 함께 동승한 형제자매 및 배우자의 수를 의미 (단, 본인은 포함되지 않음.)
#parch: 타이타닉호에 함께 동승한 부모님 및 자녀의 수를 의미 (단, 본인은 포함되지 않음.)
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1 #+1을 하는 이유는 본인까지 포함하기 위함


# In[10]:


#새로운 feature 확인 - head 함수 사용
titanic.head()


# ---

# #### 모델 학습시키기(Logistic Regression, Random Forest, XGBoost)

# In[11]:


#모델 학습 준비
#학습에 필요한 feature: 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’

titanic = titanic[['survived', 'pclass', 'sex', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]


# In[12]:


#feature, target 분리

#feature
#전체 데이터 가운데, 예측해야 하는 'survived' 데이터를 drop ('survived': target)
#axis=1을 이용하여 feature에서 'survived'열 삭제
X = titanic.drop('survived', axis=1)

#target
y = titanic['survived']


# ##### Logistic Regression, Random Forest, XGBoost를 통해서 생존자를 예측하는 모델을 학습

# - Logistic Regression

# In[13]:


#라이브러리 임포트
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[14]:


#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


#데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[16]:


#모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)


# In[17]:


#예측
y_pred = model.predict(X_test)


# In[18]:


#평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification_Report:\n {classification_report(y_test, y_pred)}')


# - Random Forest

# In[19]:


#라이브러리 임포트
from sklearn.tree import DecisionTreeClassifier


# In[20]:


#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


#데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


#모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[23]:


#예측
y_pred = model.predict(X_test)


# In[24]:


#평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n {classification_report(y_test,y_pred)}')


# - XGBoost

# In[25]:


#라이브러리 임포트
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[26]:


#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


#데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[28]:


#XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# In[29]:


#모델 학습
xgb_model.fit(X_train_scaled, y_train)


# In[30]:


#예측
y_pred_xgb = xgb_model.predict(X_test_scaled)


# In[31]:


#평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')


# In[ ]:




