CNN for Image Classification.ipynb is a project that focuses on building, training, and testing a Convolutional Neural Network (CNN) for image classification. we have 3 folders (training_set, test_set and single_prediction) in training set and test set folders we have 2 folders which inclodes dogs and cats pictures. we have to make a model with getting help from these folders to predict that two pictures in single_prediction folders are dog or cat.
Importing Necessary Libraries we have to import essential libraries for creating and training the CNN:
**tensorflow** The main framework used to build and train the CNN.
**ImageDataGenerator** A tool for preprocessing and augmenting images during training.
**numpy** For numerical operations.
**image** For loading and processing individual images for prediction.
1. Data Preprocessing
   a. Training Data
   in ImageDataGenerator I used rescale in order to Normalizes pixel values to the range [0, 1] which the normal range is 0 to 255 which helps the model converge faster during training. Also Augmenting Data that Applies transformations like shear, zoom, and horizontal flips to increase dataset variety.
Directory Path Specifies the folder where training images are stored.
   b. Testing Data Prepares the testing dataset with similar rescaling but without augmentation.
2. Defining the CNN Architecture  
here Sequential Model: Defines a simple CNN in a sequential order. in next lines we will describe layers:
   a. Convolutional Layer:
filters=32: 32 feature detectors.
kernel_size=3: 3x3 kernel size.
activation='relu': Applies the ReLU activation function to introduce non-linearity.
input_shape=[64, 64, 3]: Accepts images with dimensions 64x64 and 3 color channels (RGB).
   b. Pooling Layer Reduces the spatial dimensions by taking the maximum value in 2x2 regions.
3. Compiling and Training the Model
   here optimizer='adam' Uses the Adam optimizer for training.
loss='binary_crossentropy' Specifies the loss function for binary classification.
metrics=['accuracy'] Tracks accuracy during training.
And Training Fits the model to the training data and validates it on the test data.
Runs for 5 epochs. after each epoch we can see the amount of accuracy and loss.
4 - Making a single prediction
   here firstly we load the specific image after that Converts the loaded image into a NumPy array with pixel values and then the shape of the array is (64, 64, 3) (height, width, channels).
test_image = np.expand_dims(test_image, axis=0): Adds an additional dimension to the image array so it matches the format expected by the CNN model which the new shape becomes (1, 64, 64, 3). This step is essential because the model expects a batch of images, not a single image. then result = cnn.predict(test_image) : Uses the trained CNN model to predict whether the image is a cat or a dog.
Output: result will be a probability score in a 2D array. For example:
If result[0][0] > 0.5, the image is classified as a Dog.
If result[0][0] <= 0.5, the image is classified as a Cat.
