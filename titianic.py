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