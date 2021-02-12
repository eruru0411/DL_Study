# id : 환자 식별 번호
# diagnosis : 양성여부(M=악성, B=양성)
# radius : 반경
# texture : 질감
# perimeter : 둘레
# area : 면적
# smoothness : 매끄러움
# compactness : 조그만 정도
# concavity : 오목함
# concave points : 오목한 점의 수
# symmetry : 대칭
# dimentsion : 차원
# _mean : 2~11 평균값
# _se : 12~21 표준오차
# _worst : 22~31 : 각 세포 구분들에서 제일 큰 3개의 값을 평균낸 값


# 1.EDA : 데이터 사전 탐색 (데이터 PCA 거침)
#     0) DataFrame 살펴보기
#     1) 차트그리기
# 2.결측치/이상치/Object --> 처리
# 3.feature engineering 피쳐 전처리/가공
# 4.학습/평가 : f1(불균형 데이터이므로 accuracy 사용x), auc
# 5.검증(GridSearchCV, confusion_matrix)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Binarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

import warnings
warnings.filterwarnings(action="ignore")

#1. EDA
df = pd.read_csv("data.csv")
print(df.shape) #(569, 33)
print(df.info()) #null값x, diagnosis:object (M = malignant(악성), B = benign(양성))
print(df.head()) #Unnamed:32는 전부 NaN이므로 삭제  #id: unique하므로 삭제

print(df.isnull().sum()) #Unnamed: 32 569개 결측치 -> 삭제
df = df.drop(["Unnamed: 32", "id"], axis=1) #Unnamed: 32 ,  id 삭제
print(df.shape) #(569, 31)

print(df.describe())

print(df["diagnosis"].value_counts()) #B(양성):357, M(악성):212 -> target data
print('양성', round(df['diagnosis'].value_counts()[0]/len(df) * 100,2)) #양성 62.74
print('악성', round(df['diagnosis'].value_counts()[1]/len(df) * 100,2)) #악성 37.26
sns.countplot(x="diagnosis", data=df)
#plt.show()


#Object 처리(target data)
df["diagnosis"] = df["diagnosis"].apply(lambda x:0 if x =="B" else 1) #0이면 양성, 1이면 악성
# print(pd.Series(y).value_counts())#array형이므로 reshape으로 바꿔줌
# print(y.head())

#target data 분리
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
print(X.shape), print(y.shape)



# 2.결측치/이상치/Object --> 처리

#상관분석
# f,ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# plt.show()


#이상치 -> boxplot?
def CHECK_OUTLIER(df=None, column=None):
    Q1 = np.percentile(df[column].values, 25)
    Q3 = np.percentile(df[column].values, 75)
    IQR = Q3 - Q1
    IQR_weight = IQR * 1.5
    minimum = Q1 - IQR_weight
    maximum = Q3 + IQR_weight
    outlier_idx = df[column][  (df[column]<minimum) | (df[column]>maximum)  ].index
    return outlier_idx
    outlier_idx = CHECK_OUTLIER(df=df, column=col)
    print(col, oulier_idx)

numeric_columns = df.dtypes[df.dtypes != 'object'].index
for i, col in enumerate(numeric_columns) :
    outlier_idx = CHECK_OUTLIER(df=df, column=col)
    print(col , outlier_idx)
    #df.drop(outlier_idx, axis=0, inplace=True)
# #데이터 수가 적은데 이상치가 많아서 drop시키면 데이터가 빈약해질 것 같아서 여기서는 이상치 제거 없이 진행

#모델 분석
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)

print("------{}-------".format(str))
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
auc = roc_auc_score(y_test, proba[:,1].reshape(-1,1))

print("정확도{:.4f}  F1 {:.4f}=(정밀도{:.4f}  재현률{:.4f} auc{:.4f}) ".format(acc, f1, precision, recall, auc))
cf_matrix = confusion_matrix(y_test, pred)
print(cf_matrix)

