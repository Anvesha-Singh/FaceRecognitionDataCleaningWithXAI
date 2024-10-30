# Augmenting each picture to increase dataset

from PIL import Image
from imgaug import augmenters as iaa

# Path to your LFW dataset
data_dir = r'lfw_data\lfw'
augmented_dir = r'lfw_data\aug'

# Create a directory for augmented images
os.makedirs(augmented_dir, exist_ok=True)

# Count the number of images per class
class_counts = {}
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            class_name = os.path.basename(root)
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

# Find classes with fewer than 2 instances
classes_to_augment = [cls for cls, count in class_counts.items() if count < 2]

# Augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Affine(rotate=(-30, 30)),  # Rotate images
    iaa.Multiply((0.8, 1.2)),  # Change brightness
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Add noise
])

# Process each class with fewer than 2 instances
for cls in classes_to_augment:
    class_path = os.path.join(data_dir, cls)
    images = [file for file in os.listdir(class_path) if file.endswith(('.jpg', '.png'))]

    # Create a directory for the class inside the augmented directory
    class_augmented_dir = os.path.join(augmented_dir, cls)
    os.makedirs(class_augmented_dir, exist_ok=True)
    
    # Load images and apply augmentation
    for img_file in images:
        img_path = os.path.join(class_path, img_file)
        image = Image.open(img_path)
        image_np = np.array(image)

        # Create augmented images
        for i in range(5):  # Number of augmented images per original image
            augmented_image = seq(image=image_np)
            aug_img = Image.fromarray(augmented_image)
            aug_img_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
            aug_img.save(os.path.join(augmented_dir, cls, aug_img_filename))