from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')
!unzip gdrive/My\Drive/Dataset/New_Dataset.zip

train_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Train'
valid_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Test'

IMAGE_SIZE = [224, 224]

# add preprocessing layer to the front of ResNet50
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
  layer.trainable = False

 # useful for getting number of classes
folders = glob('New_Dataset/DevanagariHandwrittenCharacterDataset/Train/*')
print(len(folders))

# our layers - you can add more if you want
x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Image Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('New_Dataset/DevanagariHandwrittenCharacterDataset/Train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('New_Dataset/DevanagariHandwrittenCharacterDataset/Test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')
# fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=25,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Evaluate the model on the test set
final_loss, final_accuracy = model.evaluate(test_set, steps=len(test_set))

# Print the final accuracy and loss
print(f'Final Loss: {final_loss:.4f}')
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# Save the model
model.save('resnet_model.h5')
