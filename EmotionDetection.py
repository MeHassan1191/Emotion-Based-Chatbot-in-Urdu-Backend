# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords
#
# # Load the dataset
# data = pd.read_csv('C:\\Users\\hp\\Downloads\\Train.csv')
#
# # Preprocessing function for Urdu text
# def preprocess(text):
#     # Normalize text
#     text = text.lower()
#     # Remove punctuations and numbers
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\d+', '', text)
#     # Tokenization and removing stop words
#     stop_words = set("""
# اور اپنی ہے میں کی یہ تھا وہ پر کہ اس ہیں اس کے وہ بھی تھے تو ہو گا لیکن تک نے ایک سے نہیں ہوتی یا وہی جس کو میں سے کیا کرنے کیا کریں گے ان کا تھا کیا کر وہ بھی تھا ہو سکتا
# """.split())
#     words = nltk.word_tokenize(text)
#     words = [word for word in words if word not in stop_words]
#     # Joining back into a string
#     text = ' '.join(words)
#     return text
#
# # Apply preprocessing to each content
# data['content'] = data['content'].apply(preprocess)
#
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(data['content'], data['emotion'], test_size=0.2, random_state=42)
#
# # Create a TF-IDF vectorizer and an SVM classifier
# tfidf_vectorizer = TfidfVectorizer()
# svm_classifier = SVC(kernel='linear')
#
# # Train the TF-IDF vectorizer and transform the training data
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
#
# # Train the SVM classifier
# svm_classifier.fit(X_train_tfidf, y_train)
#
# # Save the model and vectorizer to disk
# joblib.dump(svm_classifier, 'emotion_classifier_model.pkl')
# joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
#
# print("Model and vectorizer have been saved to disk.")
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# Load the trained model and the vectorizer
model = joblib.load('emotion_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function for Urdu text
def preprocess(text):
    # Normalize text
    text = text.lower()
    # Remove punctuations and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenization and removing stop words
    stop_words = set("""
اور اپنی ہے میں کی یہ تھا وہ پر کہ اس ہیں اس کے وہ بھی تھے تو ہو گا لیکن تک نے ایک سے نہیں ہوتی یا وہی جس کو میں سے کیا کرنے کیا کریں گے ان کا تھا کیا کر وہ بھی تھا ہو سکتا
""".split())
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    # Joining back into a string
    text = ' '.join(words)
    return text

# Function to predict emotion of a sentence
def predict_emotion(sentence):
    # Preprocess the sentence
    preprocessed_sentence = preprocess(sentence)
    # Vectorize the sentence
    vectorized_sentence = vectorizer.transform([preprocessed_sentence])
    # Predict the emotion
    emotion = model.predict(vectorized_sentence)

    return emotion[0]

# Loop to allow continuous user input
while True:
    user_sentence = input("Enter an Urdu sentence to predict its emotion (or type 'exit' to stop): ")
    if user_sentence.lower() == 'exit':
        break
    predicted_emotion = predict_emotion(user_sentence)
    print(f"The predicted emotion is: {predicted_emotion}")

