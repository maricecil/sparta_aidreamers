#1 데이터셋 불러오기
import seaborn as sns

titanic = sns.load_dataset('titanic')

#2 feature 분석
#2-1
titanic.head()   #()안에 원하는 숫자를 넣어 행을 결정 할수 있다.

#2-2
titanic.describe()   #수치형 값들의 통계 값들을 확인 할수있다. 그래서 나머지 컬럼들은 나오지 않는다.

#2-3
#count = 값이있는 데이터의 수.  age의 경우 714개인걸로 확인되는데 그만큼의 결측치가 존재한다.
#mean = 평균값을 나타낸다.
#std = 표준편차를 나타낸다. 평균을 기준으로 얼마나 분포 되어있는지 나타낸다 . 값이 작을수록 밀집도가 높다.
#min = 가장 최소값을 나타낸다.
#25% = 데이터의 25%의 값. 
#50% = 데이터의 50%의 값. 
#75% = 데이터의 75%의 값.
#max = 가장 최대값을 나타낸다.
#가장 std가 높은 fare의 경우 mean은 32, min 0 , max는 512로 데이터의 값의 폭이 크다고 예측할수 있으나, 75%의 값이 31인것으로 보아
#비싼 요금을 내고 탄 승객은 극히 일부이며 , 대부분은 낮은 요금으로 탑승 했다는 것을 대략적으로 추측할수 있다.

#2-4

print(titanic.isnull()) 
#결측치를 확인하는 방법은 isnull()을 사용하는데, 결측치를 True로 나타내준다. 
#그래서 Sum()을 이용 그 true 값들을 더해서 보다 쉽게 실행시킨다.
print(titanic.isnull().sum()) #sum() 을 이용하여, true값을 더해줬고 , 확인하면 describe() 에서도 확인한것처럼 age에 결측치 177개를 확인하였다.
#177+ describe count 의 age 값 714 를 더하면 891이 나온다.

#3-1 결측치 처리
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
#titanic['age'].fillna(titanic['age'].median(),inplace=True) 의 inplace가 현재 내 pandas 버전 업데이트로 인해 작동하지 않는다.
#그래서 .으로 이루어진 체인 할당이 아니라 age에 직접 할당하는 방식으로 변경하였고 정상 작동한다.
#이후 궁금한점이 하나 발생했는데 .mode는 최빈값을 찾는 메서드로 알고있다. 그런데 최빈값으로 embarked의 결측치를 바꿔주기만 하는데 
#왜?? [0]을 쓰지 ? 라는 궁금증이었고, 이유는 mode 메서드는 series 형태로 결과를 반환하며 , 이는 여러 개의 최빈값이 존재할 가능성을 있기때문이라고한다.
#그럼? 두번째로 낮은 최빈값으로 결측치를 바꿔주려면 1를쓰면 되나 ? 해서 넣어봤지만 에러가 났고 이유는 최빈값이 하나이기때문에 에러가 날수있다고한다.

#value_counts = titanic['embarked'].value_counts()
#second_mode = value_counts.index[1]


#그럼 2번째로 낮은 최빈값으로 결측치를 바꾸려면 value_counts()라는 메서드를 사용하게 되는데 이 메서드는 각 값들이 데이터에서 몇번 등장했는지
#빈도를 계산하여 내림차순으로 반환해주는 메서드이다.

#value_counts

#확인해보면 S가 646 C가 168 Q가 77로 등장했고 , 두번째 최빈값은 C인것까지 확인.

#second_mode = titanic['embarked'].value_counts(dropna=False).index[1]
#이후 값을 확인했고 , 결측값은 dropna를 사용해 제외, 이후 index[1] 2번째 값을 선택하였다. 그후 실행하니 C가 반환된걸 확인했다.

titanic['age'].isnull().sum() #이후 isnull().sum() 사용하여 결측치의 값을 확인했고, 0으로 반환되는걸 확인.

#3-2 수치형으로 인코딩
titanic['sex'] = titanic['sex'].map({'male':0, 'female':1})     #map() 함수를 사용하여 쉽게 완료
titanic['alive'] = titanic['alive'].map({'no' : 0 , 'yes' : 1})    
titanic['embarked'] = titanic['embarked'].map({'S' : 2, 'C' : 0 , 'Q' : 1})
#모델을 학습시키기 위해 문자열을 수치형데이터로 변환하는 과정이 아닐까 추측.

print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())      #이후 정상적으로 수치형 데이터로 변환 된것을 확인.

# 3-3 새로운 feature 생성
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1 
#열(2차원배열에서는 특징이 된다)을 만드는 방식은 파이썬부터 워낙 많이 실습 했기에 쉽게 진행.
titanic # 특징(feature)생성과 결측치 처리도 끝났으니 젠체 데이터를 확인하여, 제대로 동작하는지 확인


#4-1  모델 학습 준비 
titanic=titanic.drop(['class','who','adult_male','deck','embark_town','alone','alive'],axis=1)
#titanic=titanic.drop('class','who','adult_male','deck','embark_town','alone',axis=1)
#필요한 열만 골라내는건 참고에 있어서 drop 메서드를 사용해 반대로 필요 없는 열들을 지워보기로했다.
#하단 코드로 실행했을땐 오류가 나왔고 이후에 내가 삭제하고 싶은 열들을 리스트 형태로 전달하여했다.

X=titanic.drop('survived',axis=1) 
y=titanic['survived']
#생존자 예측이기에 survived를 제외한 나머지 데이터를 특징으로 삼고 학습을 진행하여야 한다.
#그래서 X에는 survived를 제외한 나머지 feature=특징을 넣어주고
#y에는 target=결과값,목표변수 만을 넣어준다.


#4-2. Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train_test_split 함수의 호출 방식은 위와 같은데 훈련 데이터와 , 테스트 데이터를 무작위로 나눠주는 함수라고 생각하면 쉽다.
#test_size를 0.2로 설정했기에, 테스트세트의 비율은 20프로로 설정되고 훈련세트는 자동으로 70프로로 설정된다.
scaler = StandardScaler()
#그동안 강의를 들으면서 생겼던 궁금증은 도대체 scaler와 정규화는 차이가 무엇일까 ? 둘다 데이터 값을 0~1로 압축해주는 과정이 아닌가?
#라고 막막하게 생각했었고, 이번 과제를 기회로 알아 보았다.
#scaler는 평균이 0, 표준편차를 1로 만드는 과정이고, 정규화는 모든 데이터를 백분율로 변환하는것이다.
#예를 들자면 scaler는 상대평가, 정규화는 최저값 0 ,최고점 1이라고 생각하면 쉬울것 같다.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#train에는 훈련데이터들이 들어가 있기 때문에 , fit_함수를 사용해서 학습 한 이후에 데이터를 변환시킨다?
#궁금증 하나 , fit_transform은 학습한후에 학습한 데이터를 변환하는데 이 변환은 스케일링을 말하는거라고 현재 이해한 상태다.
#그럼 학습하기전에 미리 스케일링을 진행해야 속도나 다른 부분에서 더 도움이 되지않을까 ?
#라는 생각을 했고 코드를 수정. 오류가 계속 발생되어 , GPT의 힘을 빌려보니
#fit() 메서드는 훈련 데이터의 평균과 표준 편차를 계산해주는 메서드였다.
#그럼 또 궁금증, 후에 모델을 학습시킬때 fit메서드를 다시 사용하는데??
#이 둘은 이름은 같지만 수행하는 작업과 목적이 다르다.
#전처리 과정에서는 위와 같은 목적을 가지지만 , 학습 단계에서 fit는 데이터의 특성을 파악하고,
#모델 학습 단계에서 fit()을 사용하여 실제 모델을 학습 시키는 용도로 사용된다.

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
# 이후 로지스틱회귀모델을 생성하고
model = LogisticRegression()
model.fit(X_train, y_train) #모델을 훈련 세트에 맞추어 학습시킴.

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
#전제척인 정확도는 약 80%
#survived에 있는 0,1 두개를 예측하며, 0에 대한 예측이 더 정확하다.


#4-3 Random forest
from sklearn.tree import DecisionTreeClassifier
#random forest 모델을 만들어줌
#이후 분할 ,스케일링 부분은 상동.
model2 = DecisionTreeClassifier(random_state=42) #random_state 는 결정트리 모델의 랜덤성을 제어하는데 사용됨.
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred2)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred2)}")
#전제척인 정확도는 약 77%
#survived에 있는 0,1 두개를 예측하며, 0에 대한 예측이 더 정확하다.


#4-4 XGBoost
import xgboost as xgb
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  #분할과 스케일링 부분은 상동.

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#모델 생성부분 생성할 트리수,학습률,각 트리의 최대 깊이,결과 재현성을 위한 난수 시드의 순서
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}') 
#참고에 나온 코드를 따라 진행했을때는 , 오차값은 0.12정도로 나왔다. 이후 트리 갯수를 100으로 늘리고
#트리수를 1000개로 늘리고 트리의 최대 깊이를 8까지 늘려봤는데 ,
#내 생각과는 다르게 오히려 오차값이 더 올랐다. 아무래도 트리의 갯수가 과도하게 늘어나서 과적합이 일어난거같다.