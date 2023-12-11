from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from glob import glob
#kept dataset inside Dataset folder of drive inside New_Dataset zip file following code is to load files in colab and unzipping zip file
from google.colab import drive
drive.mount('/content/gdrive')
!unzip gdrive/My\Drive/Dataset/New_Dataset.zip

# Assuming the 'New_Dataset' folder is in the current directory
train_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Train'
valid_path = 'New_Dataset/DevanagariHandwrittenCharacterDataset/Test'

IMAGE_SIZE = [224, 224]

# Load the pre-trained ResNet50 model
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all layers in the pre-trained model
for layer in resnet.layers:
    layer.trainable = False
    
# useful for getting number of classes
folders = glob('New_Dataset/DevanagariHandwrittenCharacterDataset/Train/*')
print(len(folders))


# Create a custom model on top of the pre-trained ResNet50
x = Flatten()(resnet.output)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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
model_checkpoint = ModelCheckpoint('best_resnet_model.h5', save_best_only=True)

#generate model histroy
model.summary()

# Train the model
history = model.fit_generator(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,  # Increase the number of epochs as needed
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save('resnet_optimised_model.h5')
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

