import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data/titanic_train.csv")

#missing values
r = data.isnull().sum()

#Yaş ın yüzde kaçı null hesabı
age_null = data["Age"].isnull().sum()/data.shape[0]*100

#age çizim
ax = data["Age"].hist(bins=15,density=True,stacked=True,alpha=0.7) #alpha saydamlık derecesi 
                                                                   #stacked birden fazla veri varsa yığılmış olarak gösterir
                                                                   #density yoğunluk grafiği olarak çizer 
data["Age"].plot(kind="density")    #density yoğunluk grafiği
ax.set(xlabel="Age") 
plt.xlim(0,90)
plt.grid()               #ızgara
plt.show()


#Embarked liman bilgisinde sadece 2 eksik var
print(data["Embarked"].value_counts()/data.shape[0]*100) 
#%72 si s den binmiş o yüzden s ile doldurucaz

data2 = data.copy()
#Eksik değerleri doldurma
#Age
data2["Age"].fillna(data2["Age"].median(skipna=True),inplace=True)  #age eksik verileri fillna fonk ile ortancasıyla doldurduk

#Embarked doldurma idxmax() en fazla olanla doldur demek
data2["Embarked"].fillna(data2["Embarked"].value_counts().idxmax(),inplace=True)


#Cabin sütununda çok eksik var çıkart tamamen
data2.drop("Cabin",axis = 1,inplace=True)

r2 = data2.isnull().sum()


#Fazladan Gereksiz Değişkenler
"""SibSp : çocuk adedi
   Parch : çocuklar için ebeveyn adedi
   İki değişken birbiriyle çok alakalı ve aralarında yüksek korelasyon var
   Bu iki değişkeni tek değişkene indirgemesi yapıcaz
   Yalnız mı Seyehat ediyor adında 
   sibsp + parch = 0 ise 1 sınıfı evet yalnız seyehat ediyor
   sibsp + parch > 0 ise 0 sınıfı hayır yalnız seyehat etmiyor 
"""
#ekleme
data2["YalnizSeyehat"] = np.where((data2["SibSp"] + data2["Parch"])>0,0,1) #0 dan büyükse 0 değilse 1 sınıfı
data2.drop("SibSp",axis=1,inplace=True)
data2.drop("Parch",axis=1,inplace=True)

#Kategorik Değişkenler
data2 = pd.get_dummies(data2,columns=["Pclass","Embarked","Sex"],drop_first=True)

data2.drop("PassengerId",axis=1,inplace=True)
data2.drop("Name",axis=1,inplace=True)
data2.drop("Ticket",axis=1,inplace=True)

#EDA Exploratory Data Analysis (Görsel Veri Analizi)
col_names = data2.columns

#Age için EDA
plt.figure(figsize=(15,8))
ax = sns.kdeplot(data2["Age"][data2.Survived==1],color="green",shade=True)
sns.kdeplot(data2["Age"][data2.Survived==0],color="red",shade=True)
plt.legend(["Survived","Died"])
plt.title("Yaş için Hayatta kalma ve ölüm yoğunluk grafiği")
ax.set(xlabel="Age")
plt.xlim(-10,85)
plt.show()
""" grafik sonucu çocuklar daha fazla hayatta kalmış bence ebeveynleri onları kurtarmak için kendini feda etmiş"""


#Fare(Ücret) için EDA
plt.figure(figsize=(15,8))
ax = sns.kdeplot(data2["Fare"][data2.Survived==1],color="green",shade=True)
sns.kdeplot(data2["Fare"][data2.Survived==0],color="red",shade=True)
plt.legend(["Survived","Died"])
plt.title("Fare için Hayatta kalma ve ölüm yoğunluk grafiği")
ax.set(xlabel="Fare")
plt.xlim(-20,200)
plt.show()
"""Grafik Sonucu 30 dolar üstü ödeyenlerin çoğu daha fazla hayatta kalmış"""


#Passenger Class için EDA
sns.barplot(x="Pclass", y="Survived", data=data, color="green")
plt.show()                 
"""Grafik sonucu class 1 daha fazla hayatta kalmış yani zengin olanlar"""

#Sex için EDA
sns.barplot(x="Sex",y="Survived",data=data,color="green")
plt.show()
"""Kadınların hayatta kalma oranı çok çok daha yüksek"""

#İnput output ayrım
x = data2.drop("Survived",axis=1)
y = data2["Survived"]
cols = x.columns
#Scale
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x = mms.fit_transform(x)
#Scaler verinin türünü nummpy array yapıyor tekrar dataframe dönüştürücez
x = pd.DataFrame(x,columns=[cols])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear",random_state=0)
lr.fit(x_train,y_train)
lr_predict = lr.predict(x_test)

print(lr.predict_proba(x_test))


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,lr_predict)*100

test_data = pd.read_csv("data/titanic_test.csv")

passengerID  = test_data.iloc[:,0]

test_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
test_data["YalnizSeyehat"] = np.where((test_data["SibSp"] + test_data["Parch"])>0,0,1) #0 dan büyükse 0 değilse 1 sınıfı
test_data.drop("SibSp",axis=1,inplace=True)
test_data.drop("Parch",axis=1,inplace=True)
r22 = test_data.isnull().sum()

test_data["Age"].fillna(test_data["Age"].median(skipna=True),inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(skipna=True),inplace=True)
r23 = test_data.isnull().sum()

test_data = pd.get_dummies(test_data,columns=["Pclass","Embarked","Sex"],drop_first=True)

from sklearn.preprocessing import MinMaxScaler
mms2 = MinMaxScaler()
test_data = mms2.fit_transform(test_data)

lr2_predict = lr.predict(test_data)

df = pd.DataFrame({"PassengerId":passengerID,
                   "Survived":lr2_predict})

df.to_csv("submission2.csv",index=False)


#Confusion metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr_predict)


#PRECISION
from sklearn.metrics import precision_score
pc = precision_score(y_test, lr_predict)

#RECALL 
from sklearn.metrics import recall_score
rs = recall_score(y_test,lr_predict)

#F1 Score
from sklearn.metrics import f1_score
fs = f1_score(y_test,lr_predict)

#ROC VE AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc = roc_auc_score(y_test, lr_predict)
fpr,tpr,thresholds = roc_curve(y_test, lr_predict)
plt.plot(fpr,tpr,label="ROC Curve") #labellar kenarıda yazılıcak adlar
plt.plot([0,1],[0,1],"k--",label = "Random Guess" ) #y = x doğrusu  #k-- siyah kesikli çizgi olsun demek
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc Curve")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc="lower right")  #kenarıya çizgi isim bilgileri
plt.show()


#Log loss değeri
from sklearn.metrics import log_loss
logloss = log_loss(y_test,lr_predict)


from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
kf = StratifiedKFold(n_splits=5,shuffle = True,random_state=42)

score = cross_val_score(lr, x_train, y_train,cv=kf,scoring="accuracy")
print(score)
print(score.mean())


#solver a karar verme
solvers = ["newton-cg","lbfgs","liblinear","sag","saga"]

for solver in solvers :
    score = cross_val_score(LogisticRegression(max_iter=4000,solver=solver,random_state=42),x_train,y_train,cv=kf
                            ,scoring="accuracy")
    print(f"{solver} : ",score.mean())


#bi de knn ile deneyip onun üstünde grid search random search yapalım
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,metric="manhattan",weights="uniform")
knn.fit(x_train,y_train)
knn_predict = knn.predict(x_test)

cm_knn = confusion_matrix(y_test, knn_predict)
acc_knn = accuracy_score(y_test, knn_predict)*100
f1 = f1_score(y_test, knn_predict)*100

#hyperparameter tuning
param_gridd = {
    'n_neighbors': [1, 3, 5, 7, 9,11,13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

from sklearn.model_selection import GridSearchCV
gsc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_gridd,cv=5,verbose=1,scoring="accuracy")

gsc.fit(x_train,y_train)

print(f"En iyi Hyperparametreler : {gsc.best_params_}")
print(f"En iyi score : {gsc.best_score_}")


#sonuç 
gsc_result = pd.DataFrame(gsc.cv_results_).sort_values("mean_test_score",ascending=True)

knn_kaggle_predict = knn.predict(test_data)
df = pd.DataFrame({"PassengerId":passengerID,
                   "Survived":knn_kaggle_predict})

df.to_csv("submission_knn.csv",index=False)

#Grid_Search Logistic Regression
param_griddd = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}

from sklearn.model_selection import GridSearchCV
gsc2 = GridSearchCV(estimator=lr, param_grid=param_griddd,cv=5,verbose=1,scoring="accuracy")
gsc2.fit(x_train,y_train)

print(f"En iyi Hyperparametreler : {gsc2.best_params_}")
print(f"En iyi score : {gsc2.best_score_}")






