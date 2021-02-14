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
# 4.학습/평가 : f1(불균형 데이터이므로 accuracy 사용x) auc
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

#---------------------------------------------
#변수중요도(Feature Importances)추출
features = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(features)

#상위 5개 features 추출
top_5_features = features.keys()[:5]
print(top_5_features)

#상위 5개 feature로 모델 재분석
X_train, X_test, y_train, y_test = train_test_split(X[top_5_features], y, test_size=0.2, random_state=121, shuffle=True)
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

print("상위 5개 정확도{:.4f}  F1 {:.4f}=(정밀도{:.4f}  재현률{:.4f} auc{:.4f}) ".format(acc, f1, precision, recall, auc))
cf_matrix = confusion_matrix(y_test, pred)
print(cf_matrix)

# def CHART_PRECISION_RECALL_CURV(y_test, proba):
# precision, recall, th = precision_recall_curve(y_test, proba[:, 1])
# print(len(precision), len(recall), len(th))
# plt.plot(th, precision[:len(th)], label="precision")
# plt.plot(th, recall[:len(th)], label="recall")
# plt.xlabel("threadshold")
# plt.ylabel("precision & recall value")
# plt.legend()  # plt.legend(["precision","recall"])
# plt.grid()
# plt.show()


#def CHART_ROC_CURV(y_test, proba, auc):
fpr, tpr, th = roc_curve(y_test, proba[:, 1])
plt.plot(fpr, tpr, label='ROC')
plt.plot([0, 1], [0, 1], label='th:0.5')
auc = roc_auc_score(y_test, proba[:, 1].reshape(-1, 1))
plt.title(auc)
plt.xlabel("FPR")
plt.ylabel("TPR(recall)")
plt.grid()
plt.show()

#Cross Validation
from sklearn.model_selection import cross_validate
my_score={"acc":"accuracy", "f1":"f1"} #scoring 여러개 주는 방법
score_list = cross_validate(model, X[top_5_features], y, scoring=my_score, cv=5, verbose=0)

print("score_list----->", score_list)
score_df = pd.DataFrame(score_list)
print(score_df.head())
print("상위 5개 cross_validation 평균 정확도", score_df["test_acc"].mean())
print("상위 5개 cross_validation 평균 f1", score_df["test_f1"].mean())
my_hyper_param = { "n_estimators"      : [100, 300], # 랜덤 포레스트 안의 결정 트리 갯수 #default:100
                    "max_depth"        : [3,5,7,9], #최대 tree depth
                    "min_samples_leaf" : [1,3,5], #최소한으로 말단 노드 몇개 줄지 정함
                    "random_state"    :  [121,] #bootstrapping : 복원추출(나온번호 또 허용될 수 있음)
                } #randomforest의 parameter #랜덤포레스트가 가질 수 있는 파라미터를 여기에 지정해서 넣으면 됨

#GridSearchCV
from sklearn.model_selection import GridSearchCV
gcv_model = GridSearchCV(model, param_grid=my_hyper_param, scoring="accuracy", refit=True, cv=5, verbose=0) #refit=true: 가장 잘나온 최적의 모델을 찾아냈으면 바로 예측에 반영시켜라  #cv : 몇번 돌릴건지 #f1이 scoring해주므로 밑에 score해줄 필요 x
#제일 좋은 모델 찾아서 학습 시키고 출력해라
#predict도 필요 없음. 내부적으로 파라미터을 준 갯수만큼 가장 좋았던 모델을 여기에 던져주므로

gcv_model.fit(X_train, y_train)
print("best_estimator_", gcv_model.best_estimator_) #제일 좋은 모델
print("best_params_",     gcv_model.best_params_) #제일 좋았던 파라미터
print("best_score_",     gcv_model.best_score_) #제일 잘나온 점수
y_pred = model.predict(X_test).astype(np.int32)
