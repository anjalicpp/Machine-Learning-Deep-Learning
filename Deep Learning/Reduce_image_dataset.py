dataset
Modified_data
   |
DevanagariHandwrittenCharacterDataset
   |
   |_Test__|46 folders of images with each folder has 300 images=300*46=13800
   |_Train_|46 folders of images with each folder has 1700 imges=1700*46=78200


Inside Test of 1 single folder among 46 foldesrs C:\Users\Anjali\Desktop\Modified_data.zip\DevanagariHandwrittenCharacterDataset\Test\character_1_ka->300 images of _ka letter
Inside Train of 1 single folder among 46 foldesrs C:\Users\Anjali\Desktop\Modified_data.zip\DevanagariHandwrittenCharacterDataset\Train\character_1_ka->1700 images of _ka letter

Reduction:
removed 100 images from each 46 folder of Test folder=200*46
removed 900 images from each 46 folder of Test folder=800*46
code:
import os
import random
import zipfile

def extract_zip(zip_file, extract_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def delete_images_in_folder(folder_path, num_images_to_delete):
    # Get the list of all files in the folder
    all_images = os.listdir(folder_path)
    
    # Shuffle the list of images randomly
    random.shuffle(all_images)
    
    # Select the first 'num_images_to_delete' images to delete
    images_to_delete = all_images[:num_images_to_delete]
    
    # Construct the full path for each image
    images_to_delete_paths = [os.path.join(folder_path, image) for image in images_to_delete]
    
    # Delete the selected images
    for image_path in images_to_delete_paths:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")

def delete_images_in_all_folders(base_folder_path, num_images_to_delete_per_folder):
    # Iterate through each folder (Train or Test)
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        
        if os.path.isdir(folder_path):
            # Delete images in the folder
            delete_images_in_folder(folder_path, num_images_to_delete_per_folder)

# Example usage:
zip_file_path = "C:\\Users\\Anjali\\Desktop\\Modified_data.zip"
extracted_folder_path = "C:\\Users\\Anjali\\Desktop\\ExtractedData"
num_images_to_delete_test = 100  # Change this to the desired number of images to delete from each folder in the test folder
num_images_to_delete_train = 900  # Change this to the desired number of images to delete from each folder in the train folder

# Extract the contents of the zip file
extract_zip(zip_file_path, extracted_folder_path)

# Delete images from each folder in the test folder
test_folder_path = os.path.join(extracted_folder_path, 'DevanagariHandwrittenCharacterDataset', 'Test')
delete_images_in_all_folders(test_folder_path, num_images_to_delete_test)

# Delete images from each folder in the train folder
train_folder_path = os.path.join(extracted_folder_path, 'DevanagariHandwrittenCharacterDataset', 'Train')
delete_images_in_all_folders(train_folder_path, num_images_to_delete_train)
----------------------------
code to check images in each folder:
import os

def count_images_in_class(class_folder_path):
    # Get the list of all files in the class folder
    all_images = os.listdir(class_folder_path)
    
    # Count the number of images
    num_images = len(all_images)
    
    return num_images

def count_images_in_all_classes(base_folder_path):
    # Create a dictionary to store the counts for each class folder
    class_counts = {}
    
    # Iterate through each class folder inside Train and Test
    for dataset_folder in os.listdir(base_folder_path):
        dataset_folder_path = os.path.join(base_folder_path, dataset_folder)
        
        if os.path.isdir(dataset_folder_path):
            # Create a dictionary to store the counts for each class folder inside Train or Test
            class_counts[dataset_folder] = {}
            
            # Iterate through each class folder
            for class_folder in os.listdir(dataset_folder_path):
                class_folder_path = os.path.join(dataset_folder_path, class_folder)
                
                # Count the number of images in the class folder
                num_images = count_images_in_class(class_folder_path)
                
                # Store the count in the dictionary
                class_counts[dataset_folder][class_folder] = num_images
    
    return class_counts

# Example usage:
extracted_folder_path ="C:\\Users\\Anjali\\Desktop\\ExtractedData"

# Count and print the number of remaining images in each class in the train folder
train_folder_path = os.path.join(extracted_folder_path, 'DevanagariHandwrittenCharacterDataset',)
train_class_counts = count_images_in_all_classes(train_folder_path)

print("Devanagari Folder:")
for dataset_folder, class_counts in train_class_counts.items():
    print(f"\n{dataset_folder}:")
    for class_folder, num_images in class_counts.items():
        print(f"Class {class_folder}: {num_images} images")



