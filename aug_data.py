import os
import cv2
import imgaug.augmenters as iaa

def augment_images(images_folder):
    # Get a list of all the files in the images folder
    images = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    
    # Define the augmentations to apply
    augmentations1 = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-25, 25)),
        iaa.GaussianBlur(sigma=(0, 1.0))
    ])
    
    augmentations2 = iaa.Sequential([
        # iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
        iaa.Multiply((0.5, 0.7)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    ])
    
    augmentations3 = iaa.Sequential([
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Multiply((1.2, 1.5), per_channel=True)
        # iaa.Crop(percent=(0, 0.1)),
        # iaa.PerspectiveTransform(scale=(0.01, 0.15))
    ])
    
    # Loop through each image
    for image_filename in images:
        print("imag one "+str(image_filename))
        # Read the image
        image = cv2.imread(os.path.join(images_folder, image_filename))
        
        # Apply the first set of augmentations and save the augmented image with a prefix
        prefix1 = 'aug1_'
        augmented1 = augmentations1.augment_image(image)
        augmented_filename1 = prefix1 + image_filename
        cv2.imwrite(os.path.join(images_folder, augmented_filename1), augmented1)
        
        # Apply the second set of augmentations and save the augmented image with a prefix
        prefix2 = 'aug2_'
        augmented2 = augmentations2.augment_image(image)
        augmented_filename2 = prefix2 + image_filename
        cv2.imwrite(os.path.join(images_folder, augmented_filename2), augmented2)
        
        # Apply the third set of augmentations and save the augmented image with a prefix
        prefix3 = 'aug3_'
        augmented3 = augmentations3.augment_image(image)
        augmented_filename3 = prefix3 + image_filename
        cv2.imwrite(os.path.join(images_folder, augmented_filename3), augmented3)

# Example usage

# Loop through each folder with names a_b, where a ranges from 1 to 13 and b ranges from 14 to 49
for a in range(1, 14):
    for b in range(14, 50):
        images_folder = f"/home/summer/samik/auvsi_perception/datasets/1080p dataset/train/cropped/{a}_{b}"
        # Check if the folder exists
        if os.path.exists(images_folder):
            augment_images(images_folder)