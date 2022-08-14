import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print("""
 ~ MEME KANSERİ TEŞHİS UYGULAMASI ~
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

model = model_olustur()
model.load_weights("model_agirliklari.h5") 



while True:
    adres = input("Jpg-Png-..... Uzantılı Görüntünün Adı :") 
    img = cv2.imread(adres)
    if(type(img)==type(None)):
        print("! Belirtilen Adreste Görüntü mevcut değildir !")
        continue
    else:
        kansermi = model.predict(img.reshape((1,)+img.shape))[0] 
        print("tahmin degeri:",kansermi)
        if(kansermi>0.5):
            print("BİLGİ: Ne Yazıkkı Verdiğiniz Bilgilerdeki Kullanıcı Kanserdir :(")
        else:
            print("BİLGİ: Tebrikler Kanser değilsiniz !")

    tekrar = input("Tekrar Görüntü Tanıtmak İster misiniz ?(e/h):")
    if(tekrar=="e"):
        print("\n ******************* \n")
        continue
    else:
        break

