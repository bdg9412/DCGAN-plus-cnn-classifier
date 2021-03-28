# -*- coding: utf-8 -*-

## 환경 설정
# Python ≥3.5 is required
import sys
# assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
# assert sklearn.__version__ >= "0.20"

# try:
#     # %tensorflow_version only exists in Colab.
# #     %tensorflow_version 2.x
#     IS_COLAB = True
# except Exception:
#     IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
# assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import cv2

# to make this output stable across runs
np.random.seed(42)
tf.random.set_random_seed(42)

## 압축하여 업로드 한 데이터 셋 압축을 풀기위한 코드
# import zipfile
# with zipfile.ZipFile('/content/drive/MyDrive/LPD_competition.zip', 'r') as existing_zip:
#     existing_zip.extractall('/content/drive/MyDrive')

## 디렉토리 호출
from os import listdir

files = listdir("/home/dongkeun/다운로드/data/train") #디렉토리 내의 파일 혹은 폴더를 리스트 형태로 반환
print('total directory:',len(files))
# for i in range(10):
#     print(files[i])

def get_images(directory):
    Images = []
    Labels = []
    train_gen_data=[]
    label = 0
    '''
    라벨은 0~999까지의 숫자이므로 range(1000)을 활용하여 0~999까지를 호출
    Labels리스트에는  상위 for문에서 label변수에 할당한 iter를 할당 --> 각 이미지 별 라벨이 된다
    인자로 전달받은 directory의 내부에 0~999폴더 속에 이미지가 있으므로 이중 for문안에서 라벨링 폴더 내 개별 이미지 파일 읽어온다
    이때 cv2의 imread를 사용한다.
    '''
    
    for labels in range(1000): #Main Directory where each class label is present as folder name.:   
        labels=str(labels)
        label=labels
        #print(labels,"\n","-"*10)
        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(32,32)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            image=image/255
            flat_image= np.ravel(image)
            train_gen_data.append(flat_image)
            Images.append(image)
            Labels.append(label)
    
    return Images,Labels,train_gen_data

Images, Labels,train_gen_data = get_images('/home/dongkeun/다운로드/data/train/') #Extract the training images from the folders.

Images_np = np.array(Images,dtype=np.float32) #converting the list of images to numpy array.
Labels_np = np.array(Labels,dtype=np.float32)
train_gen_data = np.array(train_gen_data,dtype=np.float32)

print("Shape of Images:",Images_np.shape)
print("Shape of Labels:",Labels_np.shape)
print("Shape of Labels:",train_gen_data.shape)

# Images_np=np.reshape(Images_np,(-1,32*32*3))/255
# print("Shape of Images:",Images_np.shape)

# 데이터 배치를 만들고 섞습니다.
BUFFER_SIZE = 60000
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices(Images_np).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(3072,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #print(1)
    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256) # 주목: 배치사이즈로 None이 주어집니다.
    #print(2)
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #print(3)
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #print(4)
    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model

generator = make_generator_model()
noise = tf.random.normal([1, 3072])
noise = tf.cast(noise, 'float32')
generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    #print('d')
    model = tf.keras.Sequential()
    #print(1)
    model.add(layers.Conv2D(64, (4,4), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    #print(2)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #print(3)
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
# print (decision)

# 이 메서드는 크로스 엔트로피 손실함수 (cross entropy loss)를 계산하기 위해 헬퍼 (helper) 함수를 반환합니다.
cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 15
noise_dim = 3072 # 32*32*3을 flatten!
num_examples_to_generate = 1000

# 이 시드를 시간이 지나도 재활용하겠습니다. 
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# `tf.function`이 어떻게 사용되는지 주목해 주세요.
# 이 데코레이터는 함수를 "컴파일"합니다.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    print('Enter the train step')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #print('traing')
        generated_images = generator(noise, training=True)
        #print(1)
        real_output = discriminator(images, training=True)
        #print(2)
        fake_output = discriminator(generated_images, training=True)
        #print(3)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # # GIF를 위한 이미지를 바로 생성합니다.
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                          epoch + 1,
    #                          seed)

    # 15 에포크가 지날 때마다 모델을 저장합니다.
    if (epoch + 1) % 15 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)
        generator.save_weights('../models/weights/generator.h5')
        discriminator.save_weights('../models/weights/discriminator.h5')
    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#   # 마지막 에포크가 끝난 후 생성합니다.
#   display.clear_output(wait=True)
#   generate_and_save_images(generator,
#                            epochs,
#                            seed)


# Commented out IPython magic to ensure Python compatibility.
# %%time
train(train_dataset, EPOCHS)

