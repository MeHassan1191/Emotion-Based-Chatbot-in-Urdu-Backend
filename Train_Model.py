# # Import necessary libraries
# import numpy as np
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping
# import stanza
# import nltk
# import pandas as pd
# from gensim.models import Word2Vec
# import pickle
# from tqdm import tqdm
#
# path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
# doc = pd.read_csv(path_name, encoding='utf-8')
#
# #Ensure all values in 'question' and 'answer' columns are strings
# doc['question'] = doc['question'].astype(str).fillna('')
# doc['answer'] = doc['answer'].astype(str).fillna('')
#
#
# doc['question'] = doc['question'].astype(str).fillna('')
# doc['answer'] = doc['answer'].astype(str).fillna('')
#
# print(doc.head())
#
# nltk.download('punkt')
#
# stanza.download('ur')
# nlp = stanza.Pipeline('ur', processors='tokenize,lemma')
#
# def tokenize(text):
#     doc_stanza = nlp(text)
#     tokens = [word.text for sentence in doc_stanza.sentences for word in sentence.words]
#     return tokens
#
# print("Tokenizing questions...")
# doc['tokenized_question'] = [tokenize(text) for text in tqdm(doc['question'], total=len(doc))]
# print("Tokenizing answers...")
# doc['tokenized_answer'] = [tokenize(text) for text in tqdm(doc['answer'], total=len(doc))]
#
# def lemmatize_text(tokens):
#     text = ' '.join(tokens)
#     doc_stanza = nlp(text)
#     lemmatized_text = [word.lemma for sentence in doc_stanza.sentences for word in sentence.words]
#     return lemmatized_text
#
# print("Lemmatizing questions...")
# doc['lemmatized_question'] = [lemmatize_text(tokens) for tokens in tqdm(doc['tokenized_question'], total=len(doc))]
# print("Lemmatizing answers...")
# doc['lemmatized_answer'] = [lemmatize_text(tokens) for tokens in tqdm(doc['tokenized_answer'], total=len(doc))]
#
# input_texts = [' '.join(tokens) for tokens in doc['lemmatized_question']]
# target_texts = ['<start> ' + ' '.join(tokens) + ' <end>' for tokens in doc['lemmatized_answer']]
#
# input_tokenizer = Tokenizer()
# output_tokenizer = Tokenizer(filters='')
# input_tokenizer.fit_on_texts(input_texts)
# output_tokenizer.fit_on_texts(target_texts)
# input_sequences = input_tokenizer.texts_to_sequences(input_texts)
# output_sequences = output_tokenizer.texts_to_sequences(target_texts)
#
# max_input_length = max(len(seq) for seq in input_sequences)
# max_output_length = max(len(seq) for seq in output_sequences)
# input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
# output_sequences = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')
#
# input_vocab_size = len(input_tokenizer.word_index) + 1
# output_vocab_size = len(output_tokenizer.word_index) + 1
#
# lemmatized_texts = doc['lemmatized_question'].tolist() + doc['lemmatized_answer'].tolist()
# word2vec_model = Word2Vec(sentences=lemmatized_texts, vector_size=100, window=5, min_count=1, workers=4)
# embedding_size = 100
#
# vocab_size = len(word2vec_model.wv.key_to_index) + 1
# embedding_matrix = np.zeros((vocab_size, embedding_size))
# for word, i in word2vec_model.wv.key_to_index.items():
#     embedding_matrix[i] = word2vec_model.wv[word]
#
# embedding_dim = 64
# lstm_units = 128
#
# encoder_inputs = Input(shape=(max_input_length,))
# encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
# encoder_lstm = LSTM(lstm_units, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
# encoder_states = [state_h, state_c]
#
# decoder_inputs = Input(shape=(None,))
# decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
# decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
# decoder_dense = Dense(output_vocab_size, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
#
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
#
# model.summary()
#
# output_sequences = np.expand_dims(output_sequences, -1)
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=100)
#
# print("Training the model...")
# model.fit([input_sequences, output_sequences], output_sequences,
#     batch_size=16,
#     epochs=100,
#     validation_split=0.1,
#     callbacks=[early_stopping]
#     )
#
# model.save('seq2seq_model_saved_model_2.keras')
#
# with open('input_tokenizer_2.pkl', 'wb') as f:
#     pickle.dump(input_tokenizer, f)
# with open('output_tokenizer_2.pkl', 'wb') as f:
#     pickle.dump(output_tokenizer, f)
# word2vec_model.save('word2vec_model_2.bin')

# import pandas as pd
# from collections import Counter
#
# # Read the CSV file into a DataFrame
# path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
# doc = pd.read_csv(path_name, encoding='utf-8')
#
# # Ensure all values in 'question' and 'answer' columns are strings
# doc['question'] = doc['question'].astype(str).fillna('')
# doc['answer'] = doc['answer'].astype(str).fillna('')
#
#
#
# def find_best_answer(input_text, df):
#     input_words = input_text.split()
#     best_match_row = None
#     max_match_count = 0
#
#     for index, row in df.iterrows():
#         question = row['question']
#         question_words = question.split()
#
#         # Count matching words
#         match_count = sum((Counter(input_words) & Counter(question_words)).values())
#
#         # Update best match
#         if match_count > max_match_count:
#             max_match_count = match_count
#             best_match_row = row['answer']
#
#     return best_match_row
#
#
# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter your question (type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             break
#
#         best_answer = find_best_answer(user_input, doc)
#
#         if best_answer:
#             print("Best matching answer:", best_answer)
#         else:
#             print("No suitable answer found.")



#                       Train Simple Model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Read the CSV file into a DataFrame
path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
doc = pd.read_csv(path_name, encoding='utf-8')

# Ensure all values in 'question' and 'answer' columns are strings
doc['question'] = doc['question'].astype(str).fillna('')
doc['answer'] = doc['answer'].astype(str).fillna('')

# Set up stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words and remove stopwords, then lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Preprocess all questions and answers
doc['processed_question'] = doc['question'].apply(preprocess_text)
doc['processed_answer'] = doc['answer'].apply(preprocess_text)

# Vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(doc['processed_question']).toarray()

# Convert answers to numerical labels
unique_answers = doc['processed_answer'].unique()
answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
y = np.array([answer_to_index[answer] for answer in doc['processed_answer']])
y = to_categorical(y, num_classes=len(unique_answers))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a more complex neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(unique_answers), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1500, batch_size=512, validation_data=(X_test, y_test))

# Save the model
model.save('question_answer_model.keras')

# Load the model
loaded_model = load_model('question_answer_model.keras')


def find_best_answer(input_text, model, vectorizer):
    # Preprocess the input text
    input_text = preprocess_text(input_text)
    # Transform the input text to a TF-IDF vector
    input_vector = vectorizer.transform([input_text]).toarray()
    # Predict the most likely answers
    predictions = model.predict(input_vector)

    # Get indices of answers sorted by prediction confidence
    sorted_indices = np.argsort(predictions[0])[::-1]

    # Find the shortest valid answer with highest confidence
    for idx in sorted_indices:
        answer = unique_answers[idx]
        if answer in answer_to_index:
            return answer

    return None


if __name__ == "__main__":
    while True:
        user_input = input("اپنا سوال درج کریں (خروج کے لیے 'exit' لکھیں): ")
        if user_input.lower() == 'exit':
            break

        best_answer = find_best_answer(user_input, loaded_model, vectorizer)

        if best_answer:
            print("بہترین ملتا جلتا جواب:", best_answer)
        else:
            print("کوئی مناسب جواب نہیں ملا.")

#                      Train Angry Model
# import pandas as pd
# import re
# import numpy as np
# import json
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.utils import to_categorical
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # Read the CSV file into a DataFrame
# path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
# doc = pd.read_csv(path_name, encoding='utf-8')
#
# # Filter the DataFrame to include only rows where a_emotion is 'Angry'
# doc = doc[doc['A_emotion'] == 'Angry']
#
# # Ensure all values in 'question' and 'answer' columns are strings
# doc['question'] = doc['question'].astype(str).fillna('')
# doc['answer'] = doc['answer'].astype(str).fillna('')
#
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     return text
#
# # Preprocess all questions and answers
# doc['processed_question'] = doc['question'].apply(preprocess_text)
# doc['processed_answer'] = doc['answer'].apply(preprocess_text)
#
# # Save questions and answers to a JSON file
# qa_data = {
#     'questions': doc['processed_question'].tolist(),
#     'answers': doc['processed_answer'].tolist()
# }
#
# with open('qa_data_angry.json', 'w', encoding='utf-8') as f:
#     json.dump(qa_data, f, ensure_ascii=False, indent=4)
#
# # Vectorize the questions
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(doc['processed_question']).toarray()
#
# # Convert answers to numerical labels
# unique_answers = doc['processed_answer'].unique()
# answer_to_index = {answer: idx for idx, answer in enumerate(unique_answers)}
# y = np.array([answer_to_index[answer] for answer in doc['processed_answer']])
# y = to_categorical(y, num_classes=len(unique_answers))
#
# # Split the data into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Build a neural network model
# model = Sequential()
# model.add(Input(shape=(X_train.shape[1],)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(len(unique_answers), activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
#
# # Save the model
# model.save('question_answer_model_angry.keras')



