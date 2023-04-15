import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
    df=pd.read_csv("spam.csv",encoding="latin")
    df.head()
  df.info
  df.isna().sum()
    df.rename({"v1":"label,"v2":"text"},inplace=True,axis=1)
  df.tail()
from sklearn.preprocessing import LabelEncoder
le=label Encoder()
df['label']=le fit_transform(df['label'])
  from sklearn.model.selection import train_test_split
  x_train,x_test,y_train,y_test=train_test_split(x,y,test-size=0.20,random_state=0)
print("Before OverSampling,counts of label'1':{}".format(sum(y_train==1)))
print(Before OverSampling,counts of label'0':{}\n".format(sum(y_train==0)))
from imblearn.over_sapling import SMOTE
sm=SMOTE(random_state=2)
x_train_res,y_train_res=sm.flit_resample(x_train,y_train.ravel())
print('After OverSampling,the shape of train_x:{}'.format(x_train_res_shape))  
print('After OverSampling,the shape of train_y:{}'\n.format(y_train_res_shape))
print("After OverSampling,counts of label'1':{}".format(sum(y_train_res==1)))
print("After OverSampling,counts of label'0':{}".format(sum(y_train_res==0)))
     nltk.download("stopwords")
     [nltk_data]
     [nltk_data]
     [nltk_data]
  import nltk
  from nltk.corpus import stopwords
  from nltk.stem import PorterStemmer
  import re
  corpus=[]
  length=len(df)
for i in range(0,length):
text=re.sub("[^a-zA-Z0-9]","",df["text""][i])
text=text.lower()
text=text.split()
pe PorterStemmer()
stopword=stopwords.words("english")
text=[pe.stem(word)for word in text if not word in set(stopword)]
text="".join(text)
corpus.append(text)
corpus
from sklearn.feature_extraction.text import CountVectorizer
cv=CounterVectorizer(max_featuress=35000)
x=cv.fit_transform(corpus).toarray()
  import pickel
  pickel.dump(cv,open('cv1.pkl','wb'))
df.describe()
df.shape
   df["label"].value_counts().plot(kind="bar",figsize=(12,6))
   plt.xticks(np.arange(2),('Non spam','spam'),rotation=0);   
   sc=StandardScaler()
   x_bal=sc.fit_transform(x_bal)
   x_bal=pd.DataFrame(x_bal,columns=names)
from sklearn.model_selection import train_test_split
  x_train,x_test,y_train=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.tree import DecisionTreeClassifer()
  model=DecisionTreeClassifier()
  model.fit(X_train_res,Y_train_res)
  DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
  model1=RandomForestClassifier()
  model1.fit(X_train_res,y_train_res)
  RandomForestClassifier()
from sklearn.naive_bayes import MultinomialNB
  model=MultinomialNB()  
 model.fit(X_train_res,Y_train_res)
 MultinomialNB()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
     model=Sequential()
     X_train.shape
     model.add(Dense(units=X_train_res.shape[1],activation=="relu",kernel_initializer="random_uniform"))
     model.add(Dense(units=100,activation="relu",kernel_initializer="random_uniform"))
     model.add(Dense(units=100,activatoin="reli",kernel_initializer="random_uniform"))
     model.add(Dense(units=1,activation="simoid"))
     model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
     generator=model.fit(X_train_res,Y_train_res,epochs=10,steps_per_epoch=len(X_train_res)//64)
     generator=model.(x_train_res,y_train_res,epochs=10,steps_per_epoch=len(x_train_res)//64)
  y_pred=model.predict(x_test)
  y_pred
  y_pr=np.where(y_pred>0.5,1,0)
  y_test
from sklearn.metrics import confusion_matrix,accuracy_score
 cm=confunsion_matrix(y_test,y_pr)
 score=accuracy_score(y_test,y_pr)
  print(cm)
  print('Accuracy Score Is:-',score*100)
       def new_review(new_review):
           new_review=new_review
           new_review=re.sub('[^a-zA-Z]','',new_review
           new_review=new_review.lower()
           new_review=new_review.split()
           ps=PorterStemmer()
           all_stopwords=stopwords.words('english')
           all_stopwords.remove('not')
           new_review=[ps.stem(word)for word in new_review if not word in set(all_stopwords)]
           new_review=''.join(new_review)
           new_corpus=[new_review]
           new_X_test=cv.transform(new.corpus).toarray()
           print(new_X_test)
           new_y_pred=loaded_model.predict(new_X_test)
           print(new_y_pred)
           new_X_pred=np.where(new_y_pred>0.5,1,0)
           return new_y_pred
         new_review=new_review(str(input("Enter new review...")))
from sklearn.metrcs import confusion_matrix,accuracy_score,classification_report
    cm=confusion_matrix(y_test,y_pred)
    score=accuracy_score(y_test,y_pred)
    print(cm)
    print('Accuracy Score Is Naive Bayes:-',score*100)
    cm=confusion_matrix(y_test,y_pred)
    score=accuracy_score(y_test,y_pred)
    print(cm)
    print('Accuracy Score Is:-',score1*100)
    cm1=confusion_matrix(y_test_,y_pred1)
    score1=accuracy-score(y_test,y_pred1)
    print(cm1)
    print('Accuracy Score Is:-',score1*100)
from sklearn.metrics import confusion_matrix,accuracy_score
  cm=confusion_matrix(y_test,y_pr)
  score=accuracy-score(y_test,y_pr)
  print(cm)
  print('Accuracy Score Is:-',score*100)
from sklearn.metrics import confunsion_matrix,accuracy-score
      cm=confunsion_matrix(y_test,y_pr)
      score=accuracy-score(y_test,y_pr)
    print(cm)
    print('Accuracy Score Is:-',score*100)
  model.save('spam.h5)
