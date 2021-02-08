import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


df = train_df.append(test_df)
print(df.shape)
print(df.info())
print(df.tail())


# 891번 train_df
# 892번 test
train_df = df.iloc[:891,:]
test_df = df.iloc[891:,:]




plt.figure(figsize=(10,10))
sns.heatmap(data=train_df.corr(), annot=True, fmt=".2g") #가공 전 원본 train으로 상관분석
# plt.show()


#--------------------------------------
# Target 피쳐 선정
# 1   Survived     891 non-null    int64
#--------------------------------------
X_train = train_df.drop("Survived", axis=1)             #학습용문제 : train 891개
y_train = train_df["Survived"]                          #학습용답안 : train 891개
X_test  = test_df
# X_test  = df.iloc[891:,:].drop("Survived", axis=1)    #테스트용 문제지 : 892~13??
# y_test  = None                                        #테스트용 답안지 제출  gender_submission.csv Servivied
print(X_train[:2])
print(y_train[:2])
print(X_test[:2])


#--------------------------------------
# Object 처리
#--------------------------------------
#  3   Name         891 non-null    object  --> Sex로 성별 구분 정도로 활용
#  4   Sex          891 non-null    object  --> lambda 이용해 female:0 , male:1 으로 변경
#  8   Ticket       891 non-null    object  --> 의미있는 데이터로 보기 어렵다고 판단
#  10  Cabin        204 non-null    object  --> 결측이 너무 많아 드롭(687건)
#  11  Embarked     889 non-null    object  -->

df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "female" else 1)
print(df["Sex"].head())



#C106 A51 5254 --> 글자1개만 추출  (문법공부)
df["Cabin2"] =  df["Cabin"].str[0:1]   #.str[0]

#--------------------------------------
# 결측처리 - 1.삭제   2.대체   3.예측
# 나이를 예측하기 위해 이름의 호칭 추출  SibSp	Parch
# 호칭 별 평균 나이로 Age 결측 데이터 처리
# ----------------------------------------------------------
df["Name2"] = df["Name"].str.extract("([A-Za-z]+)\.")
print("==============",df["Name2"] )

dict = df.groupby(by=["Name2"])[["Name2","Age"]].mean().astype(np.int32).to_dict()
fill_mean_func = lambda gname: gname.fillna(dict['Age'][gname.name])
df = df.groupby('Name2').apply(fill_mean_func)

print("==============", df.head())


df["Age_cate"] = df["Age"].apply(lambda x : int(x//10))
df["Embarked"] = df["Embarked"].apply(lambda x: 1 if x == "C" else (2 if x == "Q" else 3))
df["SP"] = df["SibSp"] + df["Parch"]

replace_col = ["SibSp", "Parch","Name","Name2","Age"]    #SP=SibSp+Parch     Age_cate<--Name,Name2,Age
del_col = ["Ticket","Cabin","Cabin2","Fare","Embarked"]  #Fare<--Pclass,SP   Embarked
replace_col = replace_col + del_col
df.drop(replace_col, axis=1, inplace=True)
print(df.info())


#--------------------------------------------------- df 가공이 끝났다.
train_df = df[df['PassengerId'] <= 891].copy()
test_df = df[df['PassengerId'] > 891].copy()

train_df.drop("PassengerId", axis=1, inplace=True)
test_df.drop("PassengerId", axis=1, inplace=True)

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("Survived", axis=1)

print(X_train.shape,  y_train.shape, X_test.shape)
print(X_train.head())
print(X_test.head())
print(y_train)



plt.figure(figsize=(10,10))
sns.heatmap(data=train_df.corr(), annot=True, fmt=".2f") #가공이 끝난 train으로 상관분석
#plt.show()

# -----------------------------------
# 분석 (모델선정/ 평가척도/검증)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True)


# ??모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

# fit : 학습하다
rf_model.fit(X_train, y_train)
# predict : 시험
y_pred = rf_model.predict(X_test).astype(np.int32)




my_hyper_param = {  "n_estimators"     :[100, 300] ,
                    "max_depth"        :[3,5,7,9],
                    "min_samples_leaf" :[1,3,5],
                    "random_state"     :[121,]
                 }


from sklearn.model_selection import GridSearchCV
gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring="accuracy", refit=True, cv=5, verbose=0)
#---- 이하 학습 동일 --------------------
# fit : 학습하다
gcv_model.fit(X_train, y_train)
# predict : 시험
print("best_estimator_", gcv_model.best_estimator_)
print("best_params_",    gcv_model.best_params_)
print("best_score_" ,    gcv_model.best_score_)
y_pred = gcv_model.predict(X_test).astype(np.int32)
print(y_pred)


sub_df = pd.read_csv("gender_submission.csv")
mydic = {"PassengerId": sub_df["PassengerId"],
         "Survived" : y_pred
         }
sub_df = pd.DataFrame(mydic, index=None)
#sub_df.set_index('PassengerId', inplace=True)
print(sub_df.head())
sub_df.to_csv("gender_submission22.csv", index=None)






