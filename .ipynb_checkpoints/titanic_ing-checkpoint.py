import pandas as pd
import numpy as np

# 타이타닉으로 데이터셋 불러오기
import seaborn as sns

# 데이터 로드
titanic = sns.load_dataset('titanic')

# 2_1 head 함수를 통해 데이터 프레임의 첫 5행(survived, pclass, sex, age, sibsp) 출력
print(titanic.head())
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  
3  woman       False    C  Southampton   yes  False  
4    man        True  NaN  Southampton    no   True  

# 2_2 describe 함수를 통해 기본적인 통계 확인
titanic_basic_statistics = titanic.describe()
print(titanic_basic_statistics)
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

# 2_2 describe 함수를 통해 확인할 수 있는 것에 대한 설명을 주석 또는 markdown 블록으로 설명
# count : 결측값을 제외하고 선택한 그 열의 총 데이터 수이다. 
# std : standard deviation의 약자이며, 표준편차를 말한다. 데이터의 분산이 얼마나 되었는지를 나타낸다.
# min : 선택한 그 열의 최솟값이다.
# 25% : 선택한 데이터 중 25%에 위치한다는 것
# 50% : 선택한 데이터의 중간값을 말한다.
# 70% : 선택한 데이터 중 75%에 위치하는 것 상위 30% 값이라고 볼 수 있다. (문제에서는 70, 확인한 통계에서는 75 어떤걸 써야할 지 몰라서 주어진 조건에 따랐습니다.)
# max : 선택한 그 열의 최대값이다.
# 전반적으로 나잇대가 20대 후반임을 알 수 있으며, 
# 요금 또한 max 값인 512보다 평균값인 약 32 및 각 퍼센테이지에 따른 금액 및 목록 중에 가장 높으 표준편차를 보이며,
# 이는 소수의 인원들만 높은 가격을, 이외 대부분은 평균값 약 32 이하로 지불한 것을 알 수 있다.

# 2_4 isnull함수와 sum함수를 사용해서 각 열에 대한 결측치 갯수 확인
# 각 결측치 값 + 기본적인 통계에서 확인한 값 = 891
print(titanic.isnull().sum()) 
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64

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
0
0

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
0    0
1    1
2    1
3    1
4    0
Name: sex, dtype: int64
0    0
1    1
2    1
3    1
4    0
Name: alive, dtype: int64
0    2
1    0
2    2
3    2
4    2
Name: embarked, dtype: int64

# 3-3. SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성
# 새로운 feature를 head 함수를 이용해 확인

# 동승한 자매 및 배우자 수와 동승한 부모 및 자식의 수에 나도 더해야 가족크기 값
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

#새로운 feature를 head 함수를 이용해 확인
print(titanic['family_size'].head())
0    2
1    2
2    1
3    2
4    1
Name: family_size, dtype: int64

# 4-1. 모델 학습 준비
# 학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 
# 생존자 중점으로 예측 및 feature과 target을 분리, 나머지는 feature은 필요한 것만
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
# 여기서 0.2와 42은 마치 약속한 내용인 것처럼, 자주 기준으로 쓰인다고 한다. (test는 0.2 train 0.8)

# 그리고 데이터 스케일링 진행
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 모델 생성 및 학습
# 반복 학습을 통해 기본값 200으로 두어(찾아보니 보통 200으로 둔다고 합니다) 최상의 가중치를 찾기?

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)

# 모델 정확도 평가
# 사이킷런에서 제공하는 정확도 측정 함수를 사용해서 예측예정

from sklearn.metrics import accuracy_score
accuracy = accuracy_score (y_test, y_pred)

# 평가

# classification_report 함수는 인자로 실제값(target)과 예측값(predict)을 받아서 각 클래스별로 평가지표를 계산한 후 출력해줍니다.
# 간단히 말하자면 classification_report 함수를 사용하여 분류 모델의 평가 지표를 출력한다.
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 정확도에 관련 값은 0.8044692737430168로, 약 80%의 생존을 보여준다. 0행과 1행 중 0이 보다 더 정확한 값을 보여주는 것을 알 수 있다.

# precision : 정밀도로, 예측한 클래스 중 실제로 해당 클래스인 데이터의 비율을 말한다. 0에서는 0.82, 1에서는 0.78
# Recall : 재현률로, 실제 클래스 중 예측한 클래스와 일치한 데이터의 비율을 말한다. 0에서는  0.86, 1에서는 0.73
# f1-score : Precision과 Recall의 조화평균이다.
# support: 각 클래스의 실제 데이터 수이다.
# weighted avg는 데이터 수를 고려해서 평균을 구하는 반면,
# macro avg는 각 클래스별로 동일한 비중을 둔 평균을 구하기 때문에, 클래스 데이터 수에 영향을 받지 않는다. weighted avg 값이 더 의미있는 평가 지표가 된다.

# classification_report 함수는 인자로 실제값(target)과 예측값(predict)을 받아서 각 클래스별로 평가지표를 계산한 후 출력해준다.
# 간단히 말하자면 classification_report 함수를 사용하여 분류 모델의 평가 지표를 출력한다.
Accuracy: 0.8044692737430168
Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.86      0.84       105
           1       0.78      0.73      0.76        74

    accuracy                           0.80       179
   macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179


# 4_3.  Random Forest
# 랜덤 포레스트 구현 모델 
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# 모델 학습 및 예측
rf_model.fit(x_train_scaled, y_train)
predicted_rf = rf_model.predict(x_test_scaled)

# 평가
# 주어진 문제에서는 accuracy로 풀라고 했는데 안풀리고, mean squared error로만 풀려서.. 이거 여쭤봅니다.
mse_rf = mean_squared_error(y_test, predicted_rf)
print(f'랜덤 포레스트 모델의 MSE: {mse_rf}')

print(predicted_rf)
print(y_test)
랜덤 포레스트 모델의 MSE: 0.13605153814540247
[0.08       0.11666667 0.17416667 0.99       0.32       0.89
 0.83404906 0.13066667 0.57313889 1.         0.51       0.1
 0.         0.05       0.19683333 1.         0.32       0.88767857
 0.23       0.03       0.03       0.616      0.19       0.
 0.01       0.05       0.1475     0.13       0.11       0.53
 0.01333333 0.76888889 0.81       0.68       0.30880952 0.12
 0.62       0.83404906 0.96       0.         0.03       0.19
 0.         0.12042657 0.37       0.06       0.48730952 0.
 0.27       0.41       0.9        1.         0.02       0.68
 0.3        1.         0.27166667 0.74       0.77       0.46
 0.26416667 0.9        1.         0.1935     0.12042657 1.
 0.1        0.31333333 0.26       0.9        0.85       0.94
 0.82       1.         0.01       0.02766667 0.81404906 1.
 1.         0.71       0.01       1.         1.         0.01464333
 0.35       0.33       1.         1.         0.03       0.
 0.55       0.01       0.04       0.01464333 0.         0.13333333
 0.14       0.08       1.         0.35       0.21       0.01
 1.         0.02333333 0.01       0.69       1.         0.04
 0.4        0.32666667 1.         0.05       1.         0.84
 0.165      0.005      0.41       0.25466667 0.86       0.01
 0.05       1.         1.         0.99       0.1        0.45333333
 0.93       0.47       0.14       0.         0.92589286 0.08
 0.         0.72       0.58       0.11       1.         0.5195
 0.02       0.21       0.01       1.         0.61       0.02
 0.2        0.99       0.0575     0.57       0.9        0.0725
 0.18171429 0.15       0.31       0.34       0.         0.02
 0.36436508 0.77404906 1.         0.96       0.21509524 0.46161905
 0.13666667 1.         0.1625     0.37       0.02333333 1.
 0.         0.01       0.39       0.81095238 0.84       0.54
 0.08       0.         0.02       0.93933333 0.7       ]
709    1
439    0
840    0
720    1
39     1
      ..
433    0
773    0
25     1
84     1
10     1
Name: survived, Length: 179, dtype: int64


# 4_4 XGBoost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# XGBoost 모델 생성
model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 모델 학습
model_xgb.fit(x_train_scaled, y_train)

# 예측
predicted_xgb = model_xgb.predict(x_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, predicted_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')

# 예측값 및 실제 값 비교
print(predicted_xgb)
print(y_test)

# MSE를 찾아보니 Mean Squared Error 약어이며, 회귀 모델의 성능을 평가하는 데 사용되는 지표로, 예측값과 실제값 간의 차이를 제곱하여 평균한 값이라고 한다. 
# 즉 MSE 값이 낮을수록 모델의 예측이 실제값에 가깝다. 
# 만약 다른 모델의 MSE 값에 대해 0.12 보다 높으면 XGBoost의 예측 성능이 더 좋다고 할 수 있다.
# 랜덤포레스트 MSE 값이 0.13로 나왔기에 두 모델 비교시 xGBoost 성능이 더 좋다.
XGBoost 모델의 MSE: 0.12981004899201257
[ 0.16698752  0.18724331  0.23108524  0.98317605  0.60618913  1.0965853
  0.61876315  0.10687055  0.5736484   0.9364608   0.4407758   0.09192052
  0.06697178  0.12338682  0.13861506  0.8951716   0.4407758   0.629076
  0.32881403  0.19857274  0.03834007  0.32142004  0.40133616  0.05588149
  0.10646017 -0.01246578  0.2915298   0.18724331  0.10058921  0.52024543
  0.03533683  0.47507992  0.36192825  0.73806745  0.06996049  0.21346638
  0.290539    0.61876315  0.9931994   0.06541751  0.20706718  0.05893874
  0.10004118  0.16218194  0.45986494  0.13976695  0.05588149  0.05588149
  0.23328573  0.39920133  0.82746875  0.9698616   0.02259778  0.65152603
  0.03558733  0.95869344  0.21624906  0.9590082   0.96916586  0.6003983
  0.11986353  0.9169304   0.91104573  0.17258163  0.16218194  0.8841185
  0.19420701  0.13655227  0.14552599  0.9727989   0.9264086   0.84465283
  0.54901236  0.9997753   0.10647158  0.15364176  0.56280655  0.96307063
  0.9038023   0.56575376  0.01864729  0.8802808   0.9919902   0.14474948
  0.3390517   0.12970862  1.0216055   1.044036    0.06788299  0.10921229
  0.51883316  0.08310841  0.27412885  0.1122128   0.06541751  0.13655227
  0.19850807  0.04809179  0.9086595   0.25957584  0.0889746   0.07719462
  0.99920845  0.0595828   0.10004118  0.24056642  0.92493165  0.23674293
  0.09348428  0.3313971   0.9135613   0.1493135   1.019929    0.31588817
  0.09002902  0.05588149  0.19044141  0.13699652  0.87847924  0.1369555
  0.12313352  1.024007    1.0529754   1.004104    0.15231675  0.3198015
  0.92809093  0.38061184  0.4841535   0.06996049  0.6549548   0.16698752
  0.04138246  0.6506307   0.4451367   0.47411793  1.0337715   0.1405025
  0.08228415  0.5317341   0.05588149  0.9345672   0.13465577  0.10850806
  0.14016545  0.95788324  0.13465577  0.06403151  0.8106036   0.09192052
  0.13024268  0.15231675  0.10575156  0.4443564   0.10004118  0.09009149
  0.35537878  0.61876315  0.93942446  0.71937656  0.23885867  0.2911432
  0.13375784  0.99204177  0.12360051  0.20996994  0.09705293  0.9135613
  0.03533683  0.07719462  0.3852775   0.87840545  0.31588817  0.70127213
  0.08977278  0.14935505  0.11032012  0.8319421   0.7134725 ]
709    1
439    0
840    0
720    1
39     1
      ..
433    0
773    0
25     1
84     1
10     1
Name: survived, Length: 179, dtype: int64


