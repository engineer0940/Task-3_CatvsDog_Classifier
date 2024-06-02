# Cat vs Dog Classification Model

## Introduction
The objective of this project is to implement a deep learning model (CNN) to classify images of cats and dogs. This task demonstrates the effectiveness of convolutional neural networks (CNNs) in identifying and differentiating between two distinct classes in images.

## Dataset Used
The dataset used for this project is the Kaggle "Dogs vs. Cats" dataset, which contains 24,961 images of dogs and cats. The dataset is split into a training set and a test set by 90%:10% ratio to evaluate the model's performance. Training set is further splitted into training and vaidation sets by 70% and 30% ratio.
## Preprocessing Steps
Effective preprocessing is crucial for improving the performance of the model. The following preprocessing steps were performed:
1. **Resizing**: All images were resized to a fixed size of 150x150 pixels to ensure uniform input dimensions for the model.
2. **Normalization**: Pixel values were normalized to the range [0, 1] by dividing by 255. This helps in speeding up the training process and improving convergence.
3. **Data Augmentation**: Techniques such as rotation by 10 degree, horizontal flipping, horizontal and vrtical shift were applied to the training images in a separate task to artificially increase the diversity of the training data and reduce overfitting. However, it was observed that data augmentation had no significant impact on the model's output as we have already sufficiently diversified data to nusre prevention of overfitting.

## Model Implemented
CNN is implemented using Nvidia Quadro P1000 4GB GPU installed in Dell Core i7 8th gen laptop.   
The implemented model has the following architecture:
- **Conv2D Layer**: 128 filters, kernel size (3, 3), activation 'relu', input shape (150, 150, 3)
- **MaxPooling2D Layer**: pool size (2, 2)
- **Conv2D Layer**: 64 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Conv2D Layer**: 128 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Conv2D Layer**: 64 filters, kernel size (3, 3), activation 'relu'
- **MaxPooling2D Layer**: pool size (2, 2)
- **Dropout Layer**: rate 0.6 to reduce overfitting
- **Flatten Layer**: Flatten the 3D output to 1D
- **Dense Layer**: 32 units, activation 'relu'
- **Dense Layer**: 8 units, activation 'relu'
- **Dense Layer**: 2 units, activation 'softmax' for binary classification
However, for data augmentation task we have skippd the dropout layer. Moreover, we have also observed the effect of L2 regularization in an example using same model. 
## Hyperparameters
The training was conducted with:
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 30-100
- **Validation Split**: 30% of the training data was used for validation

## Results
After training the model, the following results were observed without data augmentation and using a dropout layer (after 100 epochs):
- **Test Accuracy** 91.74%
- **Test Loss**: 0.3937
With data augmntation  (Training stopped at 30 epochs)
- **Test Accuracy** 89.99%
- **Test Loss**: 0.26
With L2 regularization (Model did not converge)
- **Test Accuracy** 50.06%
- **Test Loss**: 0.75
Despite the use of data augmentation, there was no significant improvement in the model's performance. This indicates that the dataset was already sufficiently diverse or that the augmentation techniques used were not effective in this context.
## Discussion
The implemented CNN model performed well in classifying images of cats and dogs. However, the lack of significant impact from data augmentation and L2 regularization suggests that these techniques does not require for this case as it is a large dataset with diversified data. However, with data augmntation model converged earlier with lesser loss function. The training and validation accuracy and loss curves showed that the model was learning effectively, with minimal signs of overfitting.

### Future Improvements
To further improve the model's performance, the following steps could be considered:
- **Transfer Learning**: Using pre-trained models like VGG16 or ResNet50 and fine-tuning them on this dataset will result in better test accuracy with minimal training time .
- **Hyperparameter Tuning**: Experimenting with different hyperparameters and optimization techniques could yield better results with L2 regularization as model did not converge with set parameters.
- **Using Callback to save training time**: Using callback method to set a threshold value for the loss or accuracy to stop the model further training to prevent overfitting and saving training time.
