This program uses a VGG16 model to get features from input images
The CNN model works as an encoder

An RNN model is trained on features and captions which works as Decoder

The Database used in this process is: Flickr 8k Dataset
link: (https://www.kaggle.com/datasets/adityajn105/flickr8k)

The datasets is divides in three parts (sorted by name):
-First 1000 images are seperated for validation(Testing)
-Second 3018 images are seperated for Testing(But not used)
-Third 1015 images are seperated for Testing(used)

Second set of images was not used as the feature extraction was taking a lot of time
Hence the third set was created

Training features for (Trainig and Validation Dataset) are cached in same directory
Every time the script is run it uses these cached features.
file names: (vgg16_features_train.npy),(vgg16_features_val.npy)

The trained model is also cached, with tokenizer.
model:(image_caption_model.keras)
tokenizer: (tokenizer.pkl)