# -*- coding: utf-8 -*-

## 환경 설정

import sys
import sklearn
import tensorflow as tf
from tensorflow import keras

# Common imports
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# to make this output stable across runs
np.random.seed(42)
tf.random.set_random_seed(42)

## 압축하여 업로드 한 데이터 셋 압축을 풀기위한 코드
# import zipfile
# with zipfile.ZipFile('/content/drive/MyDrive/LPD_competition.zip', 'r') as existing_zip:
#     existing_zip.extractall('/content/drive/MyDrive')

## 디렉토리 호출
from os import listdir

files = listdir("/home/dongkeun/다운로드/data/train")  # 디렉토리 내의 파일 혹은 폴더를 리스트 형태로 반환
print('total directory:', len(files))
# for i in range(10):
#     print(files[i])

## 이미지 로드
def get_images(directory):
    Images = []
    Labels = []
    '''
    롯데정보통신 ai 경진대회의 경우 라벨은 0~999까지의 숫자이므로 range(1000)을 활용하여 0~999까지를 호출
    Labels리스트에는  상위 for문에서 label변수에 할당한 iter를 할당 --> 각 이미지 별 라벨이 된다
    인자로 전달받은 directory의 내부에 0~999폴더 속에 이미지가 있으므로 이중 for문안에서 라벨링 폴더 내 개별 이미지 파일 읽어온다
    이때 cv2의 imread를 사용한다.
    '''

    for labels in range(1000):  # Main Directory where each class label is present as folder name.:
        # label =[0 for i in range(1000)] #클래스 개수에 해당하는 원 핫 벡터 구현(리스트)!
        # label[labels]=1 #원 핫 벡터(리스트)에서 현재 다루는 디렉토리 위치(혹은 파일의 라벨)에 1 부여!
        label=labels
        labels = str(labels)
        # print(labels,"\n","-"*10)
        for image_file in os.listdir(
                directory + labels):  # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory + labels + r'/' + image_file)  # Reading the image (OpenCV)
            image = cv2.resize(image, (32, 32))# Resize the image, Some images are different sizes. (Resizing is very Important)
            image=image/255 #imgage Normalization
            Images.append(image)
            Labels.append(label)
    return Images, Labels

Images, Labels= get_images('/home/dongkeun/다운로드/data/train/')  # Extract the training images from the folders.


# data_augmentation = keras.Sequential(
#   [
#     tf.keras.preprocessing.image.random_rotation(x,0.2)
#   ]
# )

## 이미지 증강 및 병합

img_tmp_list=[]
# label_tmp_list=[]


for iterate,images in enumerate(Images):
    # label_tmp = [0 for i in range(Images)]
    # label_tmp[iterate] = 1
    tmp_img = tf.keras.preprocessing.image.random_rotation(images,90)
    img_tmp_list.append(np.array(tmp_img,dtype=float))
    #label_tmp_list.append(np.array(label_tmp,dtype=float))

# print('500')
Images_np_gen = np.array(img_tmp_list)  # converting the list of images to numpy array.
# Labels_np_gen = np.array(label_tmp_list)

Images_np = np.array(Images,dtype=float)  # converting the list of images to numpy array.
Labels_np = np.array(Labels,dtype=float)
print(Labels_np[:5])

print("Shape of Images:", Images_np.shape)
print("Shape of Labels:", Labels_np.shape)

Images_np=np.concatenate((Images_np,Images_np_gen),axis=0)
Labels_np=np.concatenate((Labels_np,Labels_np),axis=0)
print('concatenate done!')

##모델 생성 및 훈련

num_classes=1000
model = Sequential([
  #data_augmentation,
  # x = x.reshape((1,) + x.shape)
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy']) #모델 컴파일 시 클래스 개수 부여


epochs=5
history = model.fit(
    Images_np,Labels_np,
    epochs=epochs
)

## 테스트
def get_images_test(directory):
    Images = []

    label = 0
    '''
    롯데정보통신 ai 경진대회의 경우 라벨은 0~999까지의 숫자이므로 range(1000)을 활용하여 0~999까지를 호출
    Labels리스트에는  상위 for문에서 label변수에 할당한 iter를 할당 --> 각 이미지 별 라벨이 된다
    인자로 전달받은 directory의 내부에 0~999폴더 속에 이미지가 있으므로 이중 for문안에서 라벨링 폴더 내 개별 이미지 파일 읽어온다
    이때 cv2의 imread를 사용한다.
    '''

    for image_file in os.listdir(
            directory):  # Extracting the file name of the image from Class Label folder
        image = cv2.imread(directory  + r'/' + image_file)  # Reading the image (OpenCV)
        image = cv2.resize(image, (32, 32))  # Resize the image, Some images are different sizes. (Resizing is very Important)
        image = image / 255 #imgage Normalization
        Images.append(image)

    return Images

Images_test= get_images_test('/home/dongkeun/다운로드/data/test/')  # Extract the training images from the folders.

Images_np_test = np.array(Images_test)  # converting the list of images to numpy array.

print("Shape of Images:", Images_np_test.shape)

result_list=model.predict(Images_np_test)

## 결과 기록
f = open("predcit_cnn.txt", 'w')

for i in range(len(result_list)):
    # score=tf.nn.softmax(result_list[i])
    # cl=Class[np.argmax(score)]
    cl=np.argmax(result_list[i]) #넘파이 배열 중 최대 값의 인덱스 반환 argmin의 경우 최소 값의 인덱스 반환!
    f.write(str(cl))
    f.write("\n")
f.close()