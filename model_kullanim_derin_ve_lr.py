import cv2
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def onisle(img): 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = img/255
    img = img.reshape(50*50);
    return img

print("""
 ~   Meme Kanseri Tespit Uygulaması ~
""")
def model_olustur():
    model = Sequential()
    model.add(layers.Input((50,50,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

model_derin = model_olustur()
model_derin.load_weights("model_agirliklari.h5") 

with open('lr.model', 'rb') as handle:
    model_lr = pickle.load(handle) 



while True:
    adres = input("  Görüntünün Adı nedir :") 
    img = cv2.imread(adres)
    if(type(img)==type(None)):
        print("! Belirtilen Adreste Görüntü Mevcut Değildir !")
        continue
    else:
        hangi = input(" Hangi Model ile Tanınsın ? (derin model,logistic model):")
        if(hangi == "derin model"):
            print(" BİLGİ: derin ogrenme kullanılıyor.")
            kansermi = model_derin.predict(img.reshape((1,)+img.shape))[0] 
            print(" Tahmin Degeri:",kansermi)
        elif(hangi == "logistic model"):
            print(" BİLGİ: logistic model kullanılıyor.")
            img = onisle(img)
            kansermi = model_lr.predict(img.reshape((1,)+img.shape)) 
            print(" Tahmin Degeri:",kansermi)
        else:
            print("\n \n HATA: derin model, yada logistic model yazınız \n \n")
            continue
       
        if(kansermi>0.5):
            print(" BİLGİ: Ne Yazıkkı Verdiğiniz Bilgilerdeki Kullanıcı Kanserdir :(")
        else:
            print(" BİLGİ: Tebrikler Kanser Değilsiniz !")

    tekrar = input(" Tekrar Bir Görüntü Tanıtmak İster misiniz ?(e/h):")
    if(tekrar=="e"):
        print("\n ******************* \n")
        continue
    else:
        break

