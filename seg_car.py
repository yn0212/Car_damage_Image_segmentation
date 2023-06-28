import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
#Oxford-IIIT Pets 데이터 세트를 다운 ,세분화 마스크는 버전 3+에 포함
dataset, info = tfds.load('my_10000__car3:0.1.0', with_info=True)
print(info)
print(dataset)
#  또한 이미지 색상 값은 [0,1] 범위로 정규화됩니다.
#  마지막으로 위에서 언급한 것처럼 분할 마스크의 픽셀에는 {1, 2, 3}이라는 레이블이 지정됩니다.
#  편의를 위해 세분화 마스크에서 1을 빼면 {0, 1, 2}와 같은 레이블이 생성

def normalize(input_image, input_mask):
  #텐서를 새로운 유형으로 변환합니다.
  input_image = tf.cast(input_image, tf.float32) / 255.0

  input_mask -= 1
  return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

#num_examples : 데이터 세트의 데이터 포인트 수를 반환합니다. 즉, 데이터 세트 크기
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#입력 데이터셋의 각 원소에 주어진 함수 load_image를 적용하여 새로운 데이터셋을 생성
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

#이미지를 무작위로 뒤집어 간단한 증강을 수행
class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # 둘 다 동일한 시드를 사용하므로 동일한 임의 변경을 수행합니다.
    #Keras 전처리 레이어를 데이터 증강에 사용 
    #randomflip :훈련 중에 이미지를 무작위로 뒤집는 전처리 계층
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

#입력을 일괄 처리한 후 증강을 적용하여 입력 파이프라인을 빌드
#train_image 객체에, 순서대로 위와 같은 메소드들을 실행
train_batches = (
    train_images
    # 데이터 세트를 메모리에 캐시합니다(전처리 변환을 입력에 다시 적용하지 않아도 됨).
    .cache() 
    # 네트워크에 공급되는 샘플의 무작위 순서가 항상 있도록 샘플을 섞기.
    .shuffle(BUFFER_SIZE)
    # BATCH_SIZE 크기 묶음의 배치 샘플(마지막 것은 제외, 더 작을 수 있음)
    .batch(BATCH_SIZE)
    # 영원히 반복. 즉,데이터 세트가 배치를 계속 생성하고 데이터 부족으로 종료되지 않음.
    .repeat()
    #map으로 위에서 작성한 함수를 콜백함수로 적용시켜주는 것
    #내부 원소들 전체에 해당 함수를 적용시켜주어, 여기서는 랜덤한 데이터 변형을 해준다는 것
    .map(Augment())
    #프리페치 버퍼를 추가하면 데이터 전처리와 다운스트림 계산을 중첩하여 성능을 향상.
    #in buffer_size은 Dataset.prefetch()다음 요소를 생성하는 데 걸리는 시간에만 영향을 미침.
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

#데이터세트에서 이미지 예제와 해당 마스크를 시각화
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
#데이터세트에서 이미지 예제와 해당 마스크를 시각화해 확인하는 과정

for images, masks in train_batches.take(3):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])


#모델 정의하기
#강력한 기능을 학습하고 학습 가능한 매개변수의 수를 줄이기 위해 사전 학습된 모델인 MobileNetV2를 인코더로 사용
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

#이 레이어의 활성화를 사용
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
#모델 내 레이어 가져오기
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# 특징 추출 모델 만들기
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

#레이어 고정
down_stack.trainable = False

#디코더/업샘플러는 TensorFlow 예제에서 구현된 일련의 업샘플 블록임
#upsample : 입력을 업샘플링합니다.필터 수 ,필터 크기
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # 모델을 통한 다운샘플링
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling과 skip connections 설정
  for up, skip in zip(up_stack, skips):
    x = up(x)
    #입력 목록을 연결하는 계층
    #연결 축을 제외하고 모양이 모두 동일한 텐서 목록을 입력으로 사용하고 모든 입력을 연결한 단일 텐서를 반환
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  #Conv2DTranspose :전치된 콘볼루션의 필요성은 일반적으로 일반 콘볼루션의 반대 방향으로 진행되는 변환을 사용시
  #전치된 3D 컨볼루션을 결정하는 데 사용
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)
    #학습 및 추론 기능을 사용하여 레이어를 개체로 그룹화
  return tf.keras.Model(inputs=inputs, outputs=x)

#모델 훈련
#다중 클래스 분류 문제이므로 
# from_logits 인수가 True로 설정된 tf.keras.losses.CategoricalCrossentropy 손실 함수를 사용
# 레이블은 모든 클래스의 각 픽셀에 대한 점수 벡터가 아니라 정수 스칼라이기 때문
#create_mask 함수가 하는 일 :추론을 실행할 때 픽셀에 할당된 레이블은 값이 가장 높은 채널
OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              #레이블과 예측 간의 교차 엔트로피 손실을 계산합니다.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#결과적인 모델을 찾기
#  Keras 모델을 dot 형식으로 변환하고 파일로 저장.........
tf.keras.utils.plot_model(model, show_shapes=True)

#훈련하기 전에 모델이 예측하는 것을 확인하기 위해 모델을 시험
#create_mask :추론을 실행할 때 픽셀에 할당된 레이블은 값이 가장 높은 채널임
def create_mask(pred_mask):
  #tf.math.argmax : 텐서의 축에서 가장 큰 값을 가진 인덱스를 반환
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  #추가하고 싶은 위치에 tf.newaxis를 적어 size 변경차원 변경
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

#훈련 전 예측 확인
def show_predictions(dataset=None, num=1):
  if dataset:
    #dataset.take 해당 배치를 몇 번 불러올지 정한다.
    for image, mask in dataset.take(num):
      #예측하기
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
#훈련 전 예측 확인
show_predictions()

#  모델이 훈련되는 동안 어떻게 개선되는지 관찰하는 데 사용되는 콜백 정의
# 1에폭마다 예측결과 확인하는 콜백
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 100
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
#print('제발',test_batches)
#모델 훈련하기
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches
                          #callbacks=[DisplayCallback()]
                          )    
show_predictions(None,99)
#그래프로 확인하기
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
#3장 예측
show_predictions(test_batches, 3)