


import os 
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


sinifs = []
X=[]
Y=[]
img_dos = os.listdir("veriseti")

for dosya in img_dos[:10] :
    for sinif in os.listdir("veriseti/"+dosya):
        resimler = os.listdir("veriseti/"+dosya+"/"+sinif)
        for resim in resimler:
            img = cv2.imread("veriseti/"+dosya+"/"+sinif+"/"+resim)                 
            img = cv2.resize(img,(50,50))                                             
            X.append(img)                                                             
            if(sinif=="0"):                                                     
                Y.append(0)
            else:
                Y.append(1)

print()


print(f"""
 '1' sınıfı eleman sayısı:{sum(Y)}
 '0' sınıfı eleman sayısı:{len(Y)-sum(Y)} 
""")




X_1 = [] 
x_0 = [] 

for i,x in enumerate(X):
    if(Y[i]==1):
        X_1.append(x)
    else:
        x_0.append(x)


X_1 = X_1[:min(len(X_1),len(x_0))] 
x_0 = x_0[:min(len(X_1),len(x_0))]

X_1 = np.array(X_1)
X_0 = np.array(x_0)

X =  np.concatenate((X_1,X_0),axis=0) 
Y = np.concatenate((np.ones(len(X_1),dtype=int),np.zeros(len(X_0),dtype=int)),axis=0) 


print(X.shape,Y.shape)
#lr-----------------------------------------------------------------------------
def onisle(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = img/255 
    img = img.reshape(50*50); 
    return img

X_lr = list(map(onisle,X));
X_lr=np.array(X_lr);
print("X_LR SHAPE:",X_lr.shape)
x_train,x_test,y_train,y_test = train_test_split(X_lr,Y,random_state=0,test_size=0.35)



# Lojistik Regresyon
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train) #egittik

pred = lr.predict(x_test) #test verileri ile test ettik
acc = accuracy_score(y_test,pred) 
print(f"\n\nlogistic regrasyon acc degeri : {acc}\n\n");

# Karmaşıklık Matris
cm =confusion_matrix(y_test,pred)
print(f'Lojistik Regresyon Karmaşıklık Matrisi:\n{cm}\n')

f1skorumuz = f1_score(y_test,pred, average = 'macro')
print(f'F1 Skor Değerimiz:\n {f1skorumuz}')


#KNN ------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
cm1= confusion_matrix(y_test, y_pred)
acc1= accuracy_score(y_test,y_pred)
print(f'KNN acc değeri: {acc1} \n')

print(f'KNN karmaşıklık Matrisi:\n {cm1}\n')
#----------------------------------------------------------------------------
# SVM

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')

svc.fit(x_train,y_train)
y_pred1 = svc.predict(x_test)
acc2=accuracy_score(y_test,y_pred1)
cm2 = confusion_matrix(y_test, y_pred1)

print(f'SVM acc değeri:{acc2}\n')
print(f'SVM karmaşıklık Matrisi:\n{cm2}\n')

with open('lr.model', 'wb') as handle:
    pickle.dump(lr, handle) #pickle ile modelimizi logistik modelimizi kayıt
#----------------------------------------------------------------------------
#Gaus Bayes

from sklearn.naive_bayes import GaussianNB

gsnb = GaussianNB()
gsnb.fit(x_train,y_train)

y_pred2 = gsnb.predict(x_test)
acc3 = accuracy_score(y_test,y_pred2)
cm3 = confusion_matrix(y_test,y_pred2)

print(f'Gauss acc değeri:{acc3}\n')
print(f'Gauss karmaşıklık matrisi:\n {cm3}\n')
#---------------------------------------------------------------------------
#Karar Ağaçları
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion= 'entropy', splitter ='best')
dc.fit(x_train,y_train)

y_pred3 = dc.predict(x_test)
acc4= accuracy_score(y_test, y_pred3)
cm4 = confusion_matrix(y_test,y_pred3)

print(f'Karar Ağacı acc değeri: {acc4}\n')
print(f'Karar Ağacı karmaşıklık matrisi:\n {cm4}\n')


#derin ogrenme
#--------------------------------------------------------------------


x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=0,test_size=0.35)


def model_olustur():
    model = Sequential() #dogrusal bır model
    model.add(layers.Input((50,50,3))) 
    model.add(layers.Flatten()) 
    
    model.add(layers.Dense(640,activation='relu')) 
    model.add(layers.Dense(320,activation='relu')) 
    model.add(layers.Dense(80,activation='relu')) 
    model.add(layers.Dense(1,activation='sigmoid')) 

   
    model.compile(loss="binary_crossentropy",
                  optimizer='Adam',
                  metrics=["accuracy"])

    model.summary()
    

    return model

class myCallback(tf.keras.callbacks.Callback):
        
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99 :
                print("\n%95  kesinlik tahminimiz görüldü. Döngü  durduruldu !") 
                self.model.stop_training = True
            if logs.get('loss')<0.20:
                print("\nLoss < 0.20  olduğundan eğitim seti durduruldu ! ")
                self.model.stop_training = True
callbacks= myCallback()



derin_ogrenme = model_olustur()


history = derin_ogrenme.fit(x_train,y_train,
                            validation_data=(x_test,y_test),
                            batch_size=32,epochs=30, 
                            callbacks=[callbacks])

model_degerlendirmesi = derin_ogrenme.evaluate(x_test,y_test,batch_size=42)

print(f"modelin başarısı: {model_degerlendirmesi[1]}")

derin_ogrenme.save_weights("model_agirliklari.h5") 



#grafik 

qy_pred = derin_ogrenme.predict(x_test)

fpr, tpr , _= roc_curve(y_test,y_pred)

plt.title("Roc Eğrisi")
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

epochs = len(history.history)
print(history.history.keys())
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("Gradyan Düşüşü Eğitim veri seti grafiği")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.plot(history.history["val_loss"])
plt.title("Val_Accuracy Eğitim veri seti grafiği")
plt.show()

