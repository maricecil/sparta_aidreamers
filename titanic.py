# 1. 데이터셋 불러기기
import seaborn as sns
import pandas as pd
import numpy as np


titanic = sns.load_dataset('titanic')
#2. feature 분석
#2-1 특성 확인
titanic.head()

#2-2 기본 통계
titanic.describe()

#2-3
#1.mean : 데이터의 평균값을 의미 모든 데이터를 더하고 개수로 나눈값을 애기함
#2.std : 데이터가 평균값을 얼마정도 분포들 하고있는지 나타내는 값 
#        (값이 크면 데이터가 평균을 기준으로 많이 퍼져 있고, 값이 작으면 대부분의 데이터가 평균에 가까이에 있음)
#3.min : 데이터 중 가장 작은 값
#4.25% : 데이터의 하위 25%에 해당하는 값
#5.50% : 데이터의 중간값이고 정규적으로 분포하지 않았을떄 대표적인 값을 나타내기도 함
#6.75% : 데이터 상위 25%에 해당하는 값 
#7.max : 데이터 중 가장 큰 값
titanic.dropna().describe()
# 데이터의 누락된 값을 정리

#2-4 isnull,sum 함수를 이용하여 결측치 갯수 확인
print(titanic.isnull().sum())

#3. featur engineering

#3-1. 결측치 처리
#age(나이) 열에 있는 결측치를 "중앙값으로 대체"
titanic['age'].fillna(titanic['age'].median(), inplace=True)

#embarkde(탑승지) 열에 있는 결측치를 "최빈값으로 대체"
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# 결측치 결과
# 나이 열에 남은 결측치의 개수를 출력하고 결측치가 중앙값으로 대체으로 출력값은 "0"
print(titanic['age'].isnull().sum())
# 탑승 항구 열에 남은 결측치의 개수를 출력하고 최빈값으로 결즉치를 대체 축력값은 "0"
print(titanic['embarked'].isnull().sum())

#inplacd : true로 설정하면 데이터프레임을 수정, false로 설정하면 새로운 데이터프레임을 반환
#median : 데이터의 중앙값을 반환. 데이터의 중간에 위치한 값으로 선정
#mode : 데이터에서 가장 자주 나타내는 값을 반환
#isnull: 데이터프레임에서 결측치를 True로, 그렇지 않은 값을 False로 반환
#sum(): 숫자를 모두 더하는 함수

#3-2 수치형으로 인코딩

#sex(성별)을 남자는 0, 여자는 1로 변환
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

#alive(생존여부)를 yes는 1, no는 0으로 변환
titanic['alive'] = titanic['alive'].map({'yes': 1, 'no': 0})

#Embarked(승선항구)는 'C'는 0, 'Q'는 1, 'S'는 2로 변환
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})

# "head" 함수로 변환 결과 확인
print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())

#head 함수 : 데이터프레임의 상위 5개 행을 출력해주는 함수


#3-3 새로운 feature 생성

# sibsip(자매와 배우자 수), Parch(부모와 자식의 수)
# 위 주석을 토대로 "family_size(가족크기)를 생성
# + 1은 본인을 포함 시키기 위해 더해줌

titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())


#4. 모델 학습시키기
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#4-1 모델 학습 준비
# 모델을 학습시키기 위한 데이터를 준비
# 학습에 필요한 'feature"은 survived , pclass , sex , age , sibsp , parch , fare , embarked , family_size
# feature과 target을 분리하고 데이터 스케일링을 진행

titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target

# 'x' : 생존 여부를 제외한 모델의 입력값을 으미(feature값)
# 'y' : 생존 여부를 나타냄(target값)
# drop('survived', axis=1) : 데이터에서 열  or 행을 제거하는 함수
# 인덱싱' [ ] ' : 새로운 데이터프레임을 만듬


#4. Logistic Regression, Random Fores, XGBoost를 통해서 생존자를 예층하는 모델 학습

# Logistic Regression 과 Random Fores 모델 accuracy를 통해, XGBoost는 mean squared error를 통해 test data를 예측


#4-2 Logistic Regression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#데이터를 train과 test으로 나눔
#train_test_split() 함수 : 데이터를 무작위로 나눠 학습 데이터와 테스트 데이터를 만듬
#test_size=0.2 : 데이터 중 20%를 테스트 데이터를 사용, 80%는 학습 데이터로 사용
#random_state=42 : 무작위 데이터를 분할할 떄 사용하고 랜덤 시드를 고정값으로 설정

#데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#StandardScaler() : 평균이 0이고 표준편차가 1이 되도록 데이터를 표준화
#fit_transform() : x_train에 대해 스케일링 기준을 학습 후에 변환을 동시에 적용
#transform() : 학습한 스케일링 기준을 X_test에 동일하게 적용

#모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)
#LogisticRegression() : 회귀 모데을 생성하는 함수
#fit(): (X_train, y_train)를 사용해 모델을 학습

#예측
y_pred = model.predict(X_test)
#predict(): 입력값 X_test에 대해 예측된 타겟값(생존 여부)을 반환

#모델 평가 
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


#4-3 Random Forest

# 필요한 라이브러리 임포트
from sklearn.ensemble import RandomForestClassifier

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train_test_split: 'x'와 'y'를 무작위로 나누어 학습용,테스트용 데이터로 분활

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
#DecisionTreeClassifier: 결정 트리 분류기를 생성 데이터의 특성을 기준으로 트리 구조로 분류 수행


# 예측
y_pred = model.predict(X_test)
#predict: (X_test)에 대한 예측값을 생성 예측된 결과는 y_pred 저장

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


#4-4. XGBoost

# 필요한 라이브러리 임포트
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#XGBRegressor: XGBoost 회귀 모델을 생성
#n_estimators=100: 사용할 트리의 개수를 100으로 설정
#learning_rate=0.1: 학습률을 0.1로 설정
#max_depth=3: 각 트리의 최대 깊이를 3으로 설정
#random_state=42: 랜덤 시드를 고정하여 재현성을 보장

# 모델 학습
xgb_model.fit(X_train_scaled, y_train)

# 예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')


