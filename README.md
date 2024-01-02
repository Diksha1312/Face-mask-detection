# Face-mask-detection

### image classification task using a Convolutional Neural Network (CNN) with transfer learning (VGG19 architecture) and data augmentation

1. Data Preparation:

Images with masks (mask_path) and without masks (no_mask_path) are loaded into separate lists (img_masks and img_no_masks).
Labels for each image are appended to corresponding lists (lbl_masks and lbl_no_masks).
Two DataFrames (mask_df and no_mask_df) are created for images with masks and without masks, respectively, with columns for image paths and corresponding labels.

2. Concatenation and Shuffling:

The two DataFrames are concatenated into a single DataFrame (df).
The combined DataFrame is shuffled.

3. Data Augmentation:

An ImageDataGenerator is used for data augmentation on the training dataset. Augmentation includes rotation, width and height shifts, shear, zoom, horizontal flip, and fill mode.
Separate generators (train_ds, val_ds, and test_ds) are created for the training, validation, and test datasets, rescaling the pixel values to the range [0, 1].

4. VGG19 Model:

The VGG19 model pre-trained on ImageNet is loaded with the weights frozen.
The last layer of the VGG19 model is replaced with a new Dense layer with a single neuron and a sigmoid activation function for binary classification (mask or no mask).

5. Later the model is complied with binary crossentropy loss and RMS optimizer and trained

