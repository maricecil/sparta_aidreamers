import pandas as pd
import numpy as np

# 타이타닉으로 데이터셋 불러오기
import seaborn as sns

# 데이터 로드
titanic = sns.load_dataset('titanic')

# 2_1 head 함수를 통해 데이터 프레임의 첫 5행(survived, pclass, sex, age, sibsp) 출력
print(titanic.head())

# 2_2 describe 함수를 통해 기본적인 통계 확인
titanic_basic_statistics = titanic.describe()
print(titanic_basic_statistics)

# 2_2 describe 함수를 통해 확인할 수 있는 것에 대한 설명을 주석 또는 markdown 블록으로 설명
# count : 결측값을 제외하고 선택한 그 열의 총 데이터 수이다. 
# std : standard deviation의 약자이며, 표준편차를 말한다. 데이터의 분산이 얼마나 되었는지를 나타낸다.
# min : 선택한 그 열의 최솟값이다.
# 25% : 선택한 데이터 중 25%에 위치한다는 것
# 50% : 선택한 데이터의 중간값을 말한다.
# 70% : 선택한 데이터 중 75%에 위치하는 것 상위 30% 값이라고 볼 수 있다.
# max : 선택한 그 열의 최대값이다.
# 전반적으로 나잇대가 20대 후반임을 알 수 있으며, 
# 요금 또한 max 값인 512보다 평균값인 약 32 및 각 퍼센테이지에 따른 금액 및 목록 중에 가장 높으 표준편차를 보이며,
# 이는 소수의 인원들만 높은 가격을, 이외 대부분은 평균값 약 32 이하로 지불한 것을 알 수 있다.

# 2_4 isnull함수와 sum함수를 사용해서 각 열에 대한 결측치 갯수 확인
# 각 결측치 값 + 기본적인 통계에서 확인한 값 = 891
print(titanic.isnull().sum()) 

# age에 177개의 결측치, embarked(승선 항구)에 2개의 결측치, deck(배갑판)에서는 688개의 결측치, embarked_town에서는 2개의 결측치를 알 수 있다.

# 3_1 결측치 처리
# age의 결측치는 중앙값으로, embarked의 결측치는 최빈값으로 대체
# inplace=True으로인해 바뀐 값 타이타닉 데이터프레임에 저장


# titanic['age'].fillna(titanic['age'].median(), inplace=True) 
# 개념을 먼저 익히고 따라했는데, titanic을 통한 age를 셀렉하는 것이 아니라, 직접 age와 embarked를 셀렉 후에 작동하는 것을 알게 되었는데
# 정확한 이유를 알고 싶어서 팀원분들께 요청합니다
# when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) 

titanic.fillna({'age': titanic['age'].median()}, inplace=True) 
titanic.fillna({'embarked': titanic['embarked'].mode()[0]}, inplace=True) 

# 대체된 결과를 isnull() 함수와 sum()  함수를 print함수로 이용해서 결측치 확인
print(titanic['age'].isnull().sum())
print(titanic['embarked'].isnull().sum()) 

# 수치형으로 바꾸기 위한 작업이다.
# 3-2. Sex(성별)를 남자는 0, 여자는 1로 변환 
# alive(생존여부)를 True는 1, False는 0으로 변환 
# embarked(승선 항구)는 ‘C’는 0으로, Q는 1으로, ‘S’는 2로 변환
# 모두 변환한 후에, 변환 결과를 head 함수를 이용해 확인

# 남자는 0, 여자는 1로 인코딩
titanic['sex'] = titanic['sex'].map({'male' : 0, 'female' : 1})

# 생존여부를 True는 1, False는 0으로 인코딩
# 생존하면 1 생존하지 못하면 0
# 여기서 왜 no/yes로만 가능한지, dead나 no_alive 등 사용하면 float 값으로 변환된다. 한번 더 변환 과정을 거쳐야하기에, 수치형 사유로 인한 것이 아닌가싶다.
titanic['alive'] = titanic['alive'].map({'no': 0, 'yes': 1})


# embakred  ‘C’는 0으로, 'Q'는 1으로, ‘S’는 2로 인코딩
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 출력
print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())

# 3-3. SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성
# 새로운 feature를 head 함수를 이용해 확인

# 동승한 자매 및 배우자 수와 동승한 부모 및 자식의 수에 '나'도 더해야 가족크기 값
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

#새로운 feature를 head 함수를 이용해 확인
print(titanic['family_size'].head())

# 4-1. 모델 학습 준비
# 학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 
# feature과 target을 분리
# 그 다음 데이터 스케일링을 진행

#학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]

# 4_2. LSTM
import numpy as np
import pandas as pd
import seaborn as sns
#로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression
# 정확도 계산, 분류 보고서 생성, 혼동 행령 생성
from sklearn.metrics import classification_report, confusion_matrix
# 데이터를 훈련,테스트 분류
from sklearn.model_selection import train_test_split
# 데이터 평균을 0, 분산을 1로 스케일링
from sklearn.preprocessing import StandardScaler


# feature와 target 분리
x = titanic.drop('survived', axis=1) 
# target
y = titanic['survived'] 

# train과 test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#그리고 데이터 스케일링 진행
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# 모델 생성 및 학습
# 반복 학습을 통해 기본값 200으로 두어(찾아보니 보통 200으로 둔다고 합니다) 최상의 가중치를 찾기

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# 모델 정확도 평가
# 사이킷런에서 제공하는 정확도 측정 함수를 사용해서 예측예정

from sklearn.metrics import accuracy_score
accuracy = accuracy_score (y_test, y_pred)

# 예측
y_pred = model.predict(x_test)

# 평가

# classification_report 함수는 인자로 실제값(target)과 예측값(predict)을 받아서 각 클래스별로 평가지표를 계산한 후 출력해줍니다.
# 간단히 말하자면 classification_report 함수를 사용하여 분류 모델의 평가 지표를 출력한다.
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 정확도에 관련 값은 0.8044692737430168로, 약 80%의 생존을 보여준다.

# precision : 정밀도로, 예측한 클래스 중 실제로 해당 클래스인 데이터의 비율을 말한다. 0에서는 0.82, 1에서는 0.78
# Recall : 재현률로, 실제 클래스 중 예측한 클래스와 일치한 데이터의 비율을 말한다. 0에서는  0.86, 1에서는 0.73
# f1-score : Precision과 Recall의 조화평균이다.
# support: 각 클래스의 실제 데이터 수이다.
# weighted avg는 데이터 수를 고려해서 평균을 구하는 반면,
# macro avg는 각 클래스별로 동일한 비중을 둔 평균을 구하기 때문에, 클래스 데이터 수에 영향을 받지 않는다. weighted avg 값이 더 의미있는 평가 지표가 된다.

# classification_report 함수는 인자로 실제값(target)과 예측값(predict)을 받아서 각 클래스별로 평가지표를 계산한 후 출력해줍니다.
# 간단히 말하자면 classification_report 함수를 사용하여 분류 모델의 평가 지표를 출력한다.

# 4_3.  Random Forest
# 랜덤 포레스트 구현 모델 
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train과 test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#그리고 데이터 스케일링 진행
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# DecisionTreeRegressor을 사용해서 결정 트리 회귀 모델 생성
tree_regressor = DecisionTreeRegressor()

# 랜덤 포레스트 모델 생성
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(x_train_scaled, y_train)

# 예측
predicted_rf = rf_model.predict(x_test_scaled)

# 평가
# 주어진 문제에서는 accuracy로 풀라고 했는데 안풀리고, mean squared error로만 풀려서.. 이거 여쭤봅니다.
mse_rf = mean_squared_error(y_test, predicted_rf)
print(f'랜덤 포레스트 모델의 MSE: {mse_rf}')

print(predicted_rf)
print(y_test)

# 4_4 XGBoost
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# train과 test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#그리고 데이터 스케일링 진행
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

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

# 모델 학습
xgb_model.fit(X_train_scaled, y_train)

# 예측
predicted_xgb = xgb_model.predict(X_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, predicted_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')

print(predicted_xgb)
print(y_test)
