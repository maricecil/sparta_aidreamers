# 데이터셋 불러오기 및 feature 분석(1번, 2-1번 문제)
import seaborn as sns  

titanic = sns.load_dataset('titanic')  # 내장 데이터 로드
titanic.head()  # 첫 5행은 survived, pclass, sex, age, sibsp로 나타난다.


# describe함수를 통해 기본적인 통계 확인(2-2,3번 문제)
titanic_statistics = titanic.describe()  # 기본적인 통계 확인
min_values = titanic_statistics.loc['min']  # min : 최소값
max_values = titanic_statistics.loc['max']  # max : 최대 값
count_values = titanic_statistics.loc['count']  # count : null 값이 아닌 데이터 수
std_values = titanic_statistics.loc['std']  # std : 표준편차
q1_values = titanic_statistics.loc['25%']  # 25% : 데이터의 25%가 이 값보다 작음(1사분위)
median_values = titanic_statistics.loc['50%']  # 50% : 중앙값
q3_values = titanic_statistics.loc['75%']  # 75% : 데이터의 75%가 이 값보다 작음(3사분위) 

print(min_values)  # 각 열의 가장 작은 값은 'age'로 0.42로 보여진다. 0.42살은 약 5개월 되는 아기도 탑승했었다는 뜻이다.
print(max_values)  # 각 열의 가장 큰 값은 'fare'로 512.3292로 보여진다. 약512파운드를 지불한 승객이 있었음을 드러낸다.
print(count_values) # 빈 값이 없는 값 중 가장 큰 값은 'survived'등으로 보여진다. 891명의 생존여부가 모두 기록되어 있다는 뜻이다.
print(std_values)  # 평균으로부터 퍼져 있는 값 중 가장 큰 값은 'fare'로 49.69로 보여진다. 이는 49파운드만큼 퍼져있어 가격차가 많이 났다는 뜻이다.
print(q1_values)  # 1사분위가 가장 큰 값은 'age'로 20.1250으로 보여진다. 이는 25%의 승객이 20세 이하라는 것을 나타낸다.
print(median_values) # 중앙값이 가장 큰 값도 'age'로 28로 보여진다. 이는 50%의 승객이 28세 이하라는 것을 나타낸다. 
print(q3_values)  # 3사분위가 가장 큰 값도 'age'로 38세로 보여진다. 이는 75%의 승객이 38세 이하라는 것을 나타낸다.


# 각 열의 결측치 갯수를 확인(2-4번 문제)
print(titanic.isnull().sum())  #isnull() 함수와 sum()함수를 이용해 각 열의 결측치 갯수를 확인
# age열에 177개의 결측치가 있어 177명의 승객의 나이 정보가 누락되었음을 확인함
# deck열에 688개의 결측치가 있어 688개의 객실번호가 누락되었음을 확인함
# 이밖에 embarked와 embark_town이 탑승역(장소)이 2개씩 누락되었다.


# 결측치 처리(3-1번 문제)
titanic.fillna({'age': titanic['age'].median()}, inplace=True) 
# fillna함수를 통해 age의 결측치를 median(중앙값)으로 채웠다. inplace=true 공식으로 원본 데이터를 수정하여 age에 적용했다.
titanic.fillna({'embarked': titanic['embarked'].mode()[0]}, inplace=True) 
# fillna함수를 통해 embarked의 결측치를 mode(최빈값)으로 채웠다. inplace=true 공식으로 원본 데이터를 수정해 embarked에 적용했다.
print(titanic['age'].isnull().sum()) #age열에서 결측치가 중앙값으로 채워져 0으로 출력된다. 
print(titanic['embarked'].isnull().sum()) #embarked열에서 결측치가 최빈값(가장 많이 사용된 값)으로 채워 0으로 출력된다.


# 수치형으로 인코딩(3-2번 문제)
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
# sex열에서 male을 0, female을 1로 변환했다.
titanic['alive'] = titanic['alive'].map({'no': 0, 'yes': 1})
# alive열에서 사망한 승객을 0, 생존한 승객을 1로 변환했다.
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})
# embarked열에서 항구정보의 첫글자를 따 C,Q,S로 변환했다.
print(titanic['sex'].head())  # 처음 5개 데이터를 출력한 결과 첫번째, 다섯번째는 남성, 두번째~네번째는 여성이다.
print(titanic['alive'].head())  #첫번째, 다섯번째 남성은 사망하였고 두번째~네번째 여성은 생존하였다. 
print(titanic['embarked'].head())  # 두번째승객('0'Cherbourg)을 제외한 네 명의 승객은 '2'Southampton에서 탑승했다.


# 새로운 feature 생성(3-3번 문제)
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
# 승객의 가족크기는 형제자매/배우자(sibsp)와 부모/자녀(parch)의 수에 자신을 더해(+1) 계산한다.
print(titanic['family_size'].head())  # 즉, 다섯 승객 중 세번째, 다섯번째 승객이 혼자 탑승했고, 나머지 승객은 합승했음을 알 수 있다.  


# 모델 학습(4-1,2 Logistic Regression)
# 데이터셋 선택
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]

X = titanic.drop('survived', axis=1) # 독립변수(feature), survived를 제외한 나머지 열을 x에 저장함
y = titanic['survived'] # 종속변수, 목표(target)로 하는 survived 열을 y에 저장함


from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 함수추가
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 데이터 분할을 통해 학습데이터와 테스트데이터를 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, test는 입력데이터를, Y_는 출력데이터(생존여부)를 나타낸다.
# train_test_split 함수를 사용해 학습용(80%), 실험용(20%)로 나누었다.
# random_state는 공정성을 위해 무작위로 시드하지만 특정 숫자 42(예시)를 지정함으로써 같은 결과를 얻을 수 있다.


# 데이터 스케일링
scaler = StandardScaler()  # standerdscaler로 독립변수(feature)를 평균 0, 표준편차 1로 정규화 시킴
X_train = scaler.fit_transform(X_train)  # 학습데이터에 적용
X_test = scaler.transform(X_test)  # 테스트데이터에 적용


# 모델 생성 및 학습
model = LogisticRegression()  # 로지스틱 회귀 모델 생성
model.fit(X_train, y_train)  # 모델 학습

# 테스트 데이터를 사용해 생존 여부 예측
y_pred = model.predict(X_test)  # y_pred에 데이터 예측 결과(0 사망, 1 생존으로 저장)

# 모델 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  
# 정확도(accuracy)는 실제 값(y_test)과 예측값(y_pred)이 얼마나 일치하는지 보여줌
# 정확도 값은 0.8044로, 테스트 데이터 179명(20%) 중 80.4%의 생존 여부를 정확히 예측했다 보여짐

print(f"Classification Report:\n{classification_report(y_test, y_pred)}")  
# 정확도 외의 정밀도, 재현율, F1-score 등 성능을 상세히 보여줌
# 0번 행은 사망자를 예측하는데, 정밀도(precision)는 0.82로 82%가 사망하였음을 보여줌
# 재현율(recall)은 0.86으로 86% 확률로 사망자를 놓치지 않고 잘 찾아냈다는 것을 보여줌
# f1-점수는 정밀도와 재현율의 균형을 반영하여 84%의 정확도를 보여줌
# support에서 실제 사망자 수가 105명임을 확인함

# 마찬가지로 1번 행은 생존자를 예측한다. 
# 생존자는 74명, 정밀도 78%, 재현율 73%로 사망자 예측에 비해 성능은 떨어지지만, 76%의 정확도를 보여준다.
# 전체적으로 정확도는 80%으로, marco avg, weighted avg 모두 80%의 높은 성능을 보이고 있다.

# marco avg는 정밀도, 재현율, f1점수의 평균 값이다.
# weighted avg는더 많은 데이터가 있는 클래스(사망자)에 가중치를 주고 계산한 평균이다.


# 모델학습(4-3, Random Forest)
from sklearn.model_selection import train_test_split # 데이터 분할을 위한 함수추가
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 분할(학습용과 테스트용 으로 데이터 나눔)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링(평균 0, 표준편차 1로 정규화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 앞서 수행한 logistic 모델 생성과 같이 데이터 분할과 스케일링 과정은 동일하다.

# 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)  # decisiontree 모델 생성
model.fit(X_train, y_train)  # 모델 학습

# 예측
y_pred = model.predict(X_test)  # 생존 여부 결과(0 또는 1)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# 정확도 값은 0.7708로 77.1%의 생존 여부를 정확하게 예측한 것으로 보여짐

print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
# 여기서 사망자(0)는 105명(83%)이고, 재현율은 76% 였으며 조화 평균인 f1-점수도 80%로 높게 나와 높은 성능을 확인함
# 마찬가지로 생존자(1)는 74명(70%)이고, 재현율 78%, f1-점수 74%로 확인하였음
# 전체적으로 77%의 정확도를 보여주고 있음 


# 모델 학습(4-4, XGBoost)
from sklearn.model_selection import train_test_split # 데이터 분할을 위한 함수추가
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 앞서 수행한 logistic, decisiontree 모델 생성과 같이 데이터 분할과 스케일링 과정은 동일하다.

# XGBoost(회귀)모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# 100개의 트리를 사용해 모델을 학습하고, 0.1의 속도로 천천히, 안정적으로 학습한다.
# 각 트리의 최대 깊이는 3으로 제한해 복잡도를 제어하고, 무작위성을 42로 고정해 재현성을 보장한다.


xgb_model.fit(X_train_scaled, y_train)  # 모델 학습

# 학습된 모델을 사용해 예측을 수행함
y_pred_xgb = xgb_model.predict(X_test_scaled)  # y_pred_xgb는 예측된 생존확률 또는 점수이다.

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)  
# 예측값과 실제값 차이의 제곱을 평균낸 평균제곱오차(MSE, mean_squared_error)를 계산한다. 
# 얼마나 떨어져 있는지 알 수 있고, 값이 작을수록 모델의 예측이 실제 값과 가까움을 의미한다.
# MSE = (1/n) * Σ(actual - predicted)²

print(f'XGBoost 모델의 MSE: {mse_xgb}')
# 0.1298은 비교적 낮은 오차로 생존 여부를 예측했음을 뜻한다.
# 예측값y_pred_xgb는 실제값y_test보다 12%정도 멀어 예측이 틀릴 확률은 12%정도 임을 알 수 있다.  
# 따라서, 직관적으로 생존율, 사망율을 확인하기다는 모델을 검증하는데 사용됨을 알 수 있었다.
print(y_pred_xgb) # 생존 예측 값
print(y_test) # 실제 생존 값


# 좀 더 직관적으로 생존율과 사망율을 보기 위해서는 다음과 같이 계산한다.
import numpy as np

# 1. 예측 확률을 기준으로 0.5 이상의 값은 생존(1), 그렇지 않으면 사망(0)으로 분류
y_pred_classified = np.where(y_pred_xgb >= 0.5, 1, 0)

# 2. 전체 승객 수 계산
total_passengers = len(y_pred_classified)

# 3. 예측된 생존자 수(1)와 사망자 수(0) 계산
predicted_survived_count = np.sum(y_pred_classified == 1)  # 생존자 수
predicted_death_count = total_passengers - predicted_survived_count  # 사망자 수

# 4. 생존율과 사망률 계산
predicted_survival_rate = (predicted_survived_count / total_passengers) * 100  # 생존율
predicted_death_rate = (predicted_death_count / total_passengers) * 100  # 사망률

# 5. 결과 출력print(f"예측된 생존율: {predicted_survival_rate:.2f}%")
print(f"예측된 사망률: {predicted_death_rate:.2f}%")
print(f"예측된 생존율: {predicted_survival_rate:.2f}%")
