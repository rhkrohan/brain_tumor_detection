# Brain Tumor Detection

## Introduction

### Problem Statement

Brain tumors are **critical medical conditions** that require prompt and accurate diagnosis for effective treatment. **Traditional methods** of detecting brain tumors through manual examination of MRI scans by radiologists are often **time-consuming, prone to human error, and can delay necessary treatment**. As the volume of medical imaging data continues to grow, there is a **significant need for automated systems** to assist in **early detection** and diagnosis.

<div align="center">
    <table>
        <tr>
            <td align="center">
                <img src="https://github.com/user-attachments/assets/ca389bf9-fa8d-4ab7-941c-91950397db9a" alt="Tumor Brain" width="300"/>
                <p><b>Tumor Brain</b></p>
            </td>
            <td align="center">
                <img src="https://github.com/user-attachments/assets/b3a41b08-7f53-4dfa-a161-78b60ac35b2a" alt="Healthy Brain" width="300"/>
                <p><b>Healthy Brain</b></p>
            </td>
        </tr>
    </table>
</div>

### Project Overview

This project aims to develop an automated brain tumor detection system using **Convolutional Neural Networks (CNNs)**, a type of deep learning model particularly effective for image analysis tasks due to their ability to learn spatial hierarchies of features. The CNN architecture is designed to automatically and adaptively learn spatial hierarchies of features through **backpropagation** by utilizing multiple building blocks, such as **convolutional layers, pooling layers, and fully connected layers**. The model starts by applying various **convolutional filters** to the input MRI images to **extract low-level features** like **edges and textures**, then progresses to **higher-level features** that may represent more **complex structures** such as tumors.

The dataset used in this project consists of **brain MRI images** that have been carefully labeled as either containing a **tumor or healthy**. These images undergo a series of **preprocessing steps**, including **resizing, normalization, and data augmentation**, to enhance the quality and variability of the training data, which helps the model generalize better to new, unseen images. During training, the CNN adjusts its filters and weights through numerous iterations to minimize the classification error, learning to distinguish between subtle differences in MRI scans that **indicate the presence of a tumor**.

Additionally, advanced techniques such as dropout and **batch normalization** are employed to **prevent overfitting** and ensure robust learning. The model's performance is evaluated using metrics such as accuracy and validation accuracy on the test set to ensure it can reliably identify tumors in diverse clinical scenarios. This deep learning-based approach **leverages the power of CNNs to process and analyze medical images efficiently**, enhancing diagnostic accuracy and significantly reducing the time required for analysis, ultimately improving patient outcomes by providing timely and reliable assistance to radiologists.

## Data Preprocessing

Data preprocessing is a crucial step in preparing the MRI images for training the Convolutional Neural Network (CNN). Proper preprocessing ensures the model is trained on well-prepared data, enhancing its ability to generalize to new, unseen data. The following steps outline the preprocessing methods applied to the brain MRI images:

1. **Data Collection and Labeling:**
   - The dataset consists of MRI images of the brain, labeled into two categories: "tumor" and "healthy." These labels are essential for supervised learning, where the model learns to classify images based on the labeled data.

2. **Image Resizing:**
   - All images were resized to a uniform dimension, which is necessary for feeding images into the neural network. This step also helps reduce computational requirements and speeds up the training process. For this project, the images were resized to a standard size using TensorFlow’s preprocessing tools.

3. **Normalization:**
   - The pixel values of the MRI images were normalized to a range of 0 to 1 using TensorFlow’s preprocessing capabilities. Normalization involves scaling the pixel intensity values, which helps stabilize the learning process and improves convergence during training. This ensures that the model learns efficiently without being biased by the varying intensity scales of the raw images.

4. **Data Augmentation:**
   - Data augmentation techniques were applied to the training images using `ImageDataGenerator` from TensorFlow's Keras API. Augmentation methods included rotation, horizontal and vertical flipping, zooming, and shifting, which increase the diversity of the dataset and prevent overfitting. By simulating real-world variations in medical imaging, these techniques make the model more robust and better able to generalize to new images.

5. **Preprocessing the Training Set:**
   - The training set underwent extensive preprocessing to enhance the model’s learning capabilities. This included applying augmentation techniques such as:
     - **Rotation:** Randomly rotating images within a specified range to simulate different angles.
     - **Zooming:** Randomly zooming into images to introduce scale variations.
     - **Horizontal and Vertical Flipping:** Randomly flipping images horizontally and vertically to simulate different orientations.
     - **Shifting:** Translating images along the width and height to create spatial variance.

6. **Preprocessing the Test Set:**
   - The test set was also preprocessed to ensure consistency with the training data. While augmentation was not applied to the test set (to ensure a fair evaluation), normalization and resizing were performed to maintain uniformity across the dataset.

7. **Splitting the Dataset:**
   - The dataset was split into training, validation, and testing sets. Typically, the split ratio is around 70% for training, 15% for validation, and 15% for testing. This ensures that the model is trained on a substantial amount of data while also being evaluated on unseen images to gauge its performance.

8. **Image Batching and Shuffling:**
   - During training, images were batched and shuffled to ensure the model learns from a diverse set of images in each training epoch. Batching involves grouping a fixed number of images into batches and feeding them through the model simultaneously, which helps speed up the training process. Shuffling ensures that the model does not learn any spurious patterns related to the order of the training data.

These preprocessing steps are essential for preparing the MRI images for training a robust and accurate CNN model. By carefully preprocessing the data, the model is better equipped to learn meaningful patterns from the images, leading to improved performance in detecting brain tumors.
