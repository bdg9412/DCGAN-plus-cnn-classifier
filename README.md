# DCGAN-plus-cnn-classifier
using DCGAN for data augment and cnn classifier for classify
  
기존의 Navie GAN 구조에서 Convolution을 적용하여 성능 향상을 보인 DCGAN을 이미지 증강에 사용  
변화된 부분 요약!  
1) Discriminator - Strided Convolution 사용, Batch Normalization 사용(입력 Layer 제외), activation function: Leaky Relu
2) Generator - Transposed Convolution 사용, Batch Normalization 사용(출력 Layer 제외), activation function: Relu (마지막 Layer는 tanh)  
