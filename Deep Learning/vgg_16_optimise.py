from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout

!unzip gdrive/My\Drive/Dataset/New_Dataset.zip
train_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Train'
valid_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Test'

IMAGE_SIZE = [224, 224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all layers in the pre-trained model
for layer in vgg.layers:
    layer.trainable = False
  
   # useful for getting number of classes
folders = glob('New_Dataset/DevanagariHandwrittenCharacterDataset/Train/*')
print(len(folders))

# Create a custom model on top of the pre-trained VGG16
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the ImageDataGenerators for the training and validation sets
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

model.summary()

# Train the model
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,  # Increase the number of epochs as needed
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, model_checkpoint]
)


# Evaluate the model on the test set
final_loss, final_accuracy = model.evaluate(test_set, steps=len(test_set))

# Print the final accuracy and loss
print(f'Final Loss: {final_loss:.4f}')
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')

# Plot training history
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

# Save the final model
model.save('vgg16_optimise1_final_model.h5')



