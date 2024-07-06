import os
import numpy as np
import tensorflow
import pickle
from tqdm import tqdm

def image_loader(directory_path):
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return image_files

def extract_single_caption(img_paths, captions_file):
    captions_dict = {}
    with open(captions_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            image_name, caption = line.split(',',1)
            if image_name not in captions_dict:
                captions_dict[image_name] = caption.strip()
    captions_list= []
    for path in img_paths:
        img_name= os.path.basename(path)
        caption = captions_dict.get(img_name, "")
        captions_list.append(caption)
    return captions_list

def load_and_preprocess_image(img_path):
    t_size=(224, 224)
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=t_size) #loading image with (224,224) size
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img) #PIL to numpy array [height,width,channels]
    img_array = np.expand_dims(img_array, axis=0) #[1,height,width,channels]
    img_array = tensorflow.keras.applications.vgg16.preprocess_input(img_array) #RGB to b/w[0,255]range
    return img_array


def extract_features_vgg16(img_paths, cache_file="vgg16_features.npy"):
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        features_list = np.load(cache_file, allow_pickle=True)
        return features_list

    model = tensorflow.keras.applications.VGG16(weights='imagenet', include_top=False) #False means no fully connected layer
    features_list = []
    for img_path in tqdm(img_paths, desc="Extracting VGG16 features"):
        img_array = load_and_preprocess_image(img_path) #Preproces image
        features = model.predict(img_array) #passing image to model for feature extraction(CNN)
        features_list.append(features)
    
    features_list = np.array(features_list)
    np.save(cache_file, features_list)
    return features_list if len(features_list) > 1 else features_list[0]

def preprocess_captions(captions):
    tokenizer = tensorflow.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(seq) for seq in sequences)

    return tokenizer, vocab_size, sequences, max_length

def create_sequences(tokenizer, vocab_size, max_length, captions, features):
    X1, X2, y = [], [], []
    for img_desc, feature in zip(captions, features):
        seq = tokenizer.texts_to_sequences([img_desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = tensorflow.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = tensorflow.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
    inputs1 = tensorflow.keras.layers.Input(shape=(7, 7, 512))
    fe1 = tensorflow.keras.layers.Flatten()(inputs1)
    fe2 = tensorflow.keras.layers.Dense(256, activation='relu')(fe1)

    inputs2 = tensorflow.keras.layers.Input(shape=(max_length,))
    se1 = tensorflow.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tensorflow.keras.layers.LSTM(256)(se1)

    decoder1 = tensorflow.keras.layers.add([fe2, se2])
    decoder2 = tensorflow.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tensorflow.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    model = tensorflow.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_caption(model, tokenizer, photo_features, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tensorflow.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

def main():
    training_dir = 'Flicker Dataset\Training'
    validation_dir = 'Flicker Dataset\Validation'
    captions_file = 'Flicker Dataset\captions.txt'
    vgg16_cache_file_train = 'vgg16_features_train.npy'
    vgg16_cache_file_val = 'vgg16_features_val.npy'
    model_file = 'image_caption_model.keras'
    tokenizer_file = 'tokenizer.pkl'

    # Load image paths
    train_img_paths = image_loader(training_dir)
    val_img_paths = image_loader(validation_dir)

    # Extract single caption per image
    train_captions = extract_single_caption(train_img_paths, captions_file)
    val_captions = extract_single_caption(val_img_paths, captions_file)
    
    # Extract VGG16 features (consider batching for large datasets)
    train_features = extract_features_vgg16(train_img_paths,vgg16_cache_file_train)
    val_features = extract_features_vgg16(val_img_paths, vgg16_cache_file_val)

    # Reshape features to match the expected input shape
    train_features = np.reshape(train_features, (train_features.shape[0], 7, 7, 512))
    val_features = np.reshape(val_features, (val_features.shape[0], 7, 7, 512))

    tokenizer, vocab_size, train_sequences, train_max_length = preprocess_captions(train_captions) # Preprocess captions
    X1_train, X2_train, y_train = create_sequences(tokenizer, vocab_size, train_max_length, train_captions, train_features) # Create training sequences

    if os.path.exists(model_file) and os.path.exists(tokenizer_file):

        print("Loading model and tokenizer...")
        model = tensorflow.keras.models.load_model(model_file)
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)

        # Example of generating caption for a new image
        new_img_path = 'Flicker Dataset/Validation/10815824_2997e03d76.jpg'
        new_img_features = extract_features_vgg16([new_img_path])
        new_img_features = new_img_features.reshape((1, 7, 7, 512))
        generated_caption = generate_caption(model, tokenizer, new_img_features, train_max_length)
        from PIL import Image
        img = Image.open(new_img_path)
        img.show()
        print("Generated Caption:", generated_caption)
    else:
        model = define_model(vocab_size, train_max_length) # Define the model
        model.fit([X1_train, X2_train], y_train, epochs=20, verbose=2) # Train the model
        with open(tokenizer_file, 'wb') as f: #save the model
            pickle.dump(tokenizer, f)
        model.save(model_file)

        # Example of generating caption for a new image
        new_img_path = 'Flicker Dataset/Validation/10815824_2997e03d76.jpg'
        new_img_features = extract_features_vgg16([new_img_path])
        new_img_features = new_img_features.reshape((1, 7, 7, 512))
        generated_caption = generate_caption(model, tokenizer, new_img_features, train_max_length)
        from PIL import Image
        img = Image.open(new_img_path)
        img.show()
        print("Generated Caption:", generated_caption)

if __name__ == "__main__":
    main()