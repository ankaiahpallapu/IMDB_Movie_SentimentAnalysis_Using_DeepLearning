
# Movie Review Sentiment Analysis with CNNs

## Project Overview

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras for sentiment analysis on movie reviews. The model is trained on the IMDB movie review dataset to classify reviews as either positive or negative.

## Features

*   **Data Loading & Preprocessing**: Efficient loading of the IMDB dataset and padding sequences for uniform input length.
*   **CNN Model Architecture**: Implementation of a 1D CNN with Embedding, Conv1D, GlobalMaxPooling1D, Dense, and Dropout layers for robust feature extraction and classification.
*   **Model Training**: Training the model with early stopping to prevent overfitting.
*   **Evaluation**: Assessing model performance on unseen test data.
*   **Custom Prediction**: A utility function to predict the sentiment of any custom movie review.
*   **Visualization**: Plots for training and validation accuracy and loss over epochs.

## Technologies Used

*   **Python 3.x**
*   **TensorFlow** & **Keras**: For building and training the neural network.
*   **NumPy**: For numerical operations.
*   **Matplotlib**: For plotting and visualization.

## Setup Instructions

1.  Clone the Repository (if applicable)**:
    ```bash
    git clone <your-repo-link>
    cd movie-sentiment-cnn
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install tensorflow numpy matplotlib
    ```

4.  **Run the Jupyter Notebook / Colab Notebook**:
    Open the `movie_sentiment_analysis.ipynb` (or similar) notebook in a Jupyter environment or Google Colab and run all cells sequentially. The notebook already contains all the necessary code and explanations.

## Usage

Once the notebook cells have been executed, you can use the `predict_sentiment` function to analyze custom movie reviews:

```python
review_text = "This film was an absolute masterpiece! I was captivated from start to finish."
sentiment = predict_sentiment(review_text)
print(f"The sentiment of your review is: {sentiment}")

review_text_2 = "I found the movie incredibly boring and confusing. A waste of time."
sentiment_2 = predict_sentiment(review_text_2)
print(f"The sentiment of your review is: {sentiment_2}")
Example Output
The training process will show accuracy and loss metrics, and the final evaluation will provide overall test set performance.

Training Output Snippet:

Epoch 1/20
157/157 ━━━━━━━━━━━━━━━━━━━━ 294s 2s/step - accuracy: 0.6676 - loss: 0.5769 - val_accuracy: 0.8566 - val_loss: 0.3414
Epoch 2/20
157/157 ━━━━━━━━━━━━━━━━━━━━ 292s 2s/step - accuracy: 0.9094 - loss: 0.2401 - val_accuracy: 0.8972 - val_loss: 0.2567
...
Training completed!
Test Evaluation:

Test Loss: 0.4357
Test Accuracy: 81.20%
Custom Review Predictions:

--- TESTING WITH CUSTOM REVIEWS ---

Review 1: This movie was amazing! The acting was excellent and the story was fantastic.
Sentiment: Positive (confidence: 91.50%)

Review 2: The movie was too slow, the characters were uninteresting, and I didn't enjoy it at all.
Sentiment: Negative (confidence: 88.94%)

Review 3: The film was okay. Some parts were good, some parts were boring. I'm not sure if I would recommend it.
Sentiment: Positive (confidence: 85.36%)
Note: The model may sometimes misclassify nuanced reviews or reviews with sarcasm, as seen in Review 3 above. This is a common challenge in sentiment analysis.

Model Architecture
The model consists of:

Embedding Layer: Converts words into dense vector representations.
Conv1D Layer: Extracts local features (n-grams) from the word embeddings.
GlobalMaxPooling1D Layer: Summarizes the most important features by taking the maximum value across the sequence.
Dense Layers with Dropout: Fully connected layers for classification and regularization.
Output Layer: A sigmoid activated neuron for binary sentiment prediction.
For a detailed explanation of each layer and how the data transforms, refer to the notebook cells provided.

Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the model or add new features.

License
This project is open-source and available under the MIT License. ```

Medium Article Outline
Here's an outline for a Medium article based on this notebook, including suggested image placements:

# Building a Movie Review Sentiment Analyzer with TensorFlow and Keras

![Hero Image - Popcorn and Movie Tickets](https://via.placeholder.com/1200x600?text=Movie+Review+Sentiment+Analysis)

## Introduction: Unpacking Emotions from Text

Sentiment analysis, the process of computationally identifying and categorizing opinions expressed in a piece of text, is a fascinating application of natural language processing (NLP). In this article, we'll walk through building a simple yet effective sentiment analysis model using a Convolutional Neural Network (CNN) with TensorFlow and Keras to classify movie reviews as positive or negative.

Why CNNs for text? While often associated with image processing, 1D CNNs are excellent at identifying local patterns (like n-grams) in sequences, making them powerful for text classification tasks.

## 1. Setting Up and Loading Our Data: The IMDB Dataset

We'll use the classic IMDB movie review dataset, which comes pre-processed with Keras. It contains 50,000 movie reviews, equally split into 25,000 for training and 25,000 for testing, each labeled as positive (1) or negative (0).

To keep things manageable, we'll limit our vocabulary to the top 50,000 most frequent words.

```python
# (Code snippet for loading data and printing shapes)
import tensorflow as tf
from tensorflow import keras

num_words = 50000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

print("Training data shape (reviews):", len(x_train))
print("First 5 training labels:", y_train[:5])
Screenshot - Data Shapes Output

Decoding the Reviews
The reviews are stored as sequences of integers, where each integer represents a specific word. We can use the IMDB word index to convert these numbers back into readable text.

# (Code snippet for decoding review)
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

print("First review (as words):")
print(decode_review(x_train[0]))
Screenshot - Decoded Review

Padding Sequences
Neural networks require fixed-size inputs. Since reviews have varying lengths, we'll pad them with zeros to a maximum length (maxlen=500). Reviews shorter than maxlen will be padded, and longer ones will be truncated.

# (Code snippet for padding)
maxlen = 500
x_train_padded = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test_padded = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

print("Training data shape (padded):", x_train_padded.shape)
Screenshot - Padded Data Shapes

2. Crafting Our CNN Model Architecture
Now, let's build our CNN. The architecture is designed to capture patterns in sequences of words.

# (Code snippet for model definition)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=num_words, output_dim=256),
    keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
Screenshot - Model Summary Output

A Quick Tour of the Layers:
Embedding Layer: Translates word indices into dense vectors, capturing semantic relationships.
Conv1D Layer: Acts like a feature extractor, sliding a filter (kernel) over sequences of words to detect n-gram patterns.
GlobalMaxPooling1D Layer: Downsamples the features, picking out the most significant pattern detected by each filter across the entire review.
Dense Layers: Standard fully connected layers for learning complex combinations of features.
Dropout Layers: Regularization technique to prevent overfitting by randomly dropping neurons during training.
Output Layer: A single neuron with a sigmoid activation, outputting a probability between 0 and 1 (positive sentiment).
3. Training the Model
We compile our model with the adam optimizer, binary_crossentropy loss (suitable for binary classification), and accuracy as our metric. We also incorporate EarlyStopping to halt training when validation loss stops improving, ensuring we don't overfit.

# (Code snippet for compiling and training)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train_padded,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping]
)
Screenshot - Training Progress Output

4. Evaluating Performance
After training, we evaluate our model on the unseen test set to gauge its generalization capability.

# (Code snippet for evaluation)
test_loss, test_accuracy = model.evaluate(x_test_padded, y_test)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
Screenshot - Evaluation Output

Visualizing Training History
Plotting the training and validation accuracy and loss helps us understand the model's learning process and identify potential overfitting.

# (Code snippet for plotting history)
# ... (plotting code from notebook cell YyXSq_AhACTR)
Plot - Accuracy History Plot - Loss History

5. Predicting Sentiment for New Reviews
We've built a convenient function predict_sentiment that takes raw text, preprocesses it (converts to numbers, pads), and feeds it to our trained model to get a sentiment prediction.

# (Code snippets for text_to_numbers and predict_sentiment functions)
# ... (functions from notebook cell Jp5D4b0xADqm)

# (Code snippets for testing with custom reviews)
review1 = "This movie was amazing! The acting was excellent and the story was fantastic."
print(f"Sentiment: {predict_sentiment(review1)}\n")

review2 = "The movie was too slow, the characters were uninteresting, and I didn't enjoy it at all."
print(f"Sentiment: {predict_sentiment(review2)}\n")
Screenshot - Custom Predictions

Conclusion
We successfully built, trained, and evaluated a CNN for movie review sentiment analysis. This project showcases the power of deep learning in NLP tasks, even with relatively simple architectures. While our model achieved decent accuracy, there's always room for improvement with more complex models, larger datasets, or advanced preprocessing techniques.

Feel free to experiment with different num_words, maxlen, or model architectures to see how they impact performance
