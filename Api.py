import datetime
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, ForeignKey, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import joblib
import pyodbc
import re
import numpy as np
import json
import pandas as pd
import nltk
#from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from StanzaWorks import noun_Gender,pos_tagging,male_to_female_2nd_person,male_to_female_1nd_person,female_to_male_1nd_person,female_to_male_2nd_person
from TestDataset import find_best_answers, doc
import random as r


app = Flask(__name__)

# Define the connection string
server = 'HASSAN'
database = 'Urdu Chatbot'
username = 'sa'
password = '123'
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server'

# Create the SQLAlchemy engine
engine = create_engine(connection_string)

# Create a session maker
Session = sessionmaker(bind=engine)

# Create a base class for declarative class definitions
Base = declarative_base()

class User(Base):
    __tablename__ = '_User'

    userId = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password = Column(String(50))
    Email = Column(String(10))
    sessions = relationship("ChatSession", back_populates="user")

class Chatbot(Base):
    __tablename__ = 'Chatbot'

    id = Column(Integer, primary_key=True)
    type = Column(String(10))

class ChatSession(Base):
    __tablename__ = 'ChatSession'

    SessionId = Column(Integer, primary_key=True)
    Title = Column(String(100))
    creationDate = Column(DateTime)
    userId = Column(Integer, ForeignKey('_User.userId'))
    bot_id = Column(Integer, ForeignKey('Chatbot.id'))
    user = relationship("User", back_populates="sessions")
    chatbot = relationship("Chatbot")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = 'ChatMessage'

    MessageId = Column(Integer, primary_key=True)
    Emotion = Column(String(10))
    source = Column(String(50))
    Text = Column(String(255))
    EntryDate = Column(DateTime)
    sessionId = Column(Integer, ForeignKey('ChatSession.SessionId'))
    session = relationship("ChatSession", back_populates="messages")


path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
doc1 = pd.read_csv(path_name, encoding='utf-8')


# Ensure all values in 'question' and 'answer' columns are strings
doc1['question'] = doc['question'].astype(str).fillna('')
doc1['answer'] = doc['answer'].astype(str).fillna('')

# Load models and vectorizers for male and angry chatbots
#male_model = load_model('question_answer_model.keras')
male_vectorizer = TfidfVectorizer()
with open('qa_data.json', 'r', encoding='utf-8') as f:
    qa_data_male = json.load(f)
male_vectorizer.fit(qa_data_male['questions'])

#angry_model = load_model('question_answer_model_angry.keras')
angry_vectorizer = TfidfVectorizer()
with open('qa_data_angry.json', 'r', encoding='utf-8') as f:
    qa_data_angry = json.load(f)
angry_vectorizer.fit(qa_data_angry['questions'])

# Load model and vectorizer for emotion detection
emotion_model = joblib.load('emotion_classifier_model.pkl')
emotion_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download NLTK data
nltk.download('punkt_tab')

# Define preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_urdu_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set("""
        اور اپنی ہے میں کی یہ تھا وہ پر کہ اس ہیں اس کے وہ بھی تھے تو ہو گا لیکن تک نے ایک سے نہیں ہوتی یا وہی جس کو میں سے کیا کرنے کیا کریں گے ان کا تھا کیا کر وہ بھی تھا ہو سکتا
    """.split())
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

# Define functions to find best answers
def find_best_answer(input_text, model, vectorizer, qa_data):
    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text]).toarray()
    prediction = model.predict(input_vector)
    best_answer_index = np.argmax(prediction)
    best_answer = np.array(qa_data['answers'])[best_answer_index]
    return best_answer

def predict_emotion(sentence):
    preprocessed_sentence = preprocess_urdu_text(sentence)
    vectorized_sentence = emotion_vectorizer.transform([preprocessed_sentence])
    emotion = emotion_model.predict(vectorized_sentence)
    return emotion[0]

# def male_to_female_1st_person(sentence):
#     # Dictionary for 1st person pronoun, verb, and adjective replacements
#     replacements = {
#         "میرا": "میری",
#         "تھا": "تھی",
#         "ہوا": "ہوئی",
#         "کیا": "کی",
#         "رہا": "رہی",
#         "چاہتا": "چاہتی",
#         "سکتا": "سکتی",
#         "گیا": "گئی",
#         "لایا": "لائی",
#         "پہنچا": "پہنچی",
#         "سوچا": "سوچی",
#         "بولا": "بولی",
#         "کھایا": "کھائی",
#         "پیا": "پی",
#         "دیکھا": "دیکھی",
#         "بنا": "بنی",
#         "تھکا ہوا": "تھکی ہوئی",
#         "چلا": "چلی",
#         "لکھا": "لکھی",
#         "پڑھا": "پڑھی",
#         "لیا": "لی",
#         "دیا": "دی",
#         "بناتا": "بناتی",
#         "سنا": "سنی",
#         "بولتا": "بولتی",
#         "دیکھتا": "دیکھتی",
#         "سویا": "سوئی",
#         "سمجھا": "سمجھی",
#         "رکھا": "رکھی",
#         "اٹھایا": "اٹھائی",
#         "جاگا": "جاگی",
#         "کہا": "کہی",
#         "سنایا": "سنائی",
#         "لاتا": "لاتی",
#         "بیٹھا": "بیٹھی",
#         "پکایا": "پکائی",
#         "دوڑا": "دوڑی",
#         "پہنایا": "پہنائی",
#         "پڑھاتا": "پڑھاتی",
#         "ڈھونڈا": "ڈھونڈی",
#         "ملتا": "ملتی",
#         "لڑکا": "لڑکی",
#         "لڑکے": "لڑکیاں",
#         "آیا": "آئی",
#         "پکڑا": "پکڑی",
#         "پھینکا": "پھینکی",
#         "سیکھا": "سیکھی",
#         "گھمایا": "گھمائی",
#         "جانتا": "جانتی",
#         "کرتا": "کرتی",
#         "چکھا": "چکھی",
#         "پہنچتا": "پہنچتی",
#         "ڈرتا": "ڈرتی",
#         "رویا": "روئی",
#         "ہنسا": "ہنسی",
#         "سمجھتا": "سمجھتی"
#     }
#
#     # Tokenize the sentence into words
#     words = sentence.split()
#
#     # Replace male words with female words
#     for i, word in enumerate(words):
#         if word in replacements:
#             words[i] = replacements[word]
#
#     # Join the words back into a sentence
#     female_sentence = ' '.join(words)
#
#     return female_sentence

# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    gender = data.get('email')

    session = Session()
    if session.query(User).filter_by(username=username).first():
        session.close()
        return jsonify({'error': 'Username already taken'}), 400

    new_user = User(username=username, password=password, Email=gender)
    session.add(new_user)
    session.commit()
    session.close()

    return jsonify({'message': 'Signup successful'}), 200

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    session = Session()
    user = session.query(User).filter_by(username=username, password=password).first()
    session.close()

    if user:
        return jsonify({'message': 'Login successful'}), 200

    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/users', methods=['GET'])
def get_all_users():
    session = Session()
    users = session.query(User).all()
    user_data = [{'userId': user.userId, 'username': user.username, 'gender': user.Email} for user in users]
    session.close()
    return jsonify(user_data)

@app.route('/sessions/<username>/<int:chatbot_id>', methods=['GET'])
def get_user_sessions(username, chatbot_id):
    session = Session()
    user = session.query(User).filter_by(username=username).first()

    if not user:
        session.close()
        return jsonify({'error': 'User not found'}), 404

    user_sessions = session.query(ChatSession).filter_by(userId=user.userId, bot_id=chatbot_id).all()
    session_data = [{'sessionId': session.SessionId, 'title': session.Title, 'creationDate': session.creationDate, 'bot_id': session.bot_id}
                    for session in user_sessions]
    session.close()
    return jsonify(session_data)

@app.route('/messages/<int:session_id>', methods=['GET'])
def get_session_messages(session_id):
    session = Session()
    session_obj = session.query(ChatSession).filter_by(SessionId=session_id).first()

    if not session_obj:
        session.close()
        return jsonify({'error': 'Session not found'}), 404

    session_messages = session_obj.messages
    message_data = [{'messageId': message.MessageId, 'emotion': message.Emotion, 'text': message.Text,
                     'entryDate': message.EntryDate, 'source': message.source} for message in session_messages]
    session.close()
    return jsonify(message_data)

def save_message():
    data = request.json
    emotion = data.get('emotion')
    parent_ref = data.get('parentRef')
    source = data.get('source')
    text = data.get('text')
    entry_date = data.get('entryDate')
    session_id = data.get('sessionId')

    session = Session()
    new_message = ChatMessage(
        Emotion=emotion,
        ParentRef=parent_ref,
        source=source,
        Text=text,
        EntryDate=entry_date,
        sessionId=session_id
    )
    session.add(new_message)
    session.commit()
    session.close()
    return jsonify({'message': 'Message saved successfully'}), 200

@app.route('/savemessages', methods=['POST'])
def save_data():
    data = request.json
    emotion = data.get('emotion')
    parent_ref = data.get('parentRef')
    source = data.get('source')
    text = data.get('text')
    entry_date = data.get('entryDate')
    session_id = data.get('sessionId')
    if len(entry_date) == 26:
        entry_date = entry_date[:-3]

    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=HASSAN;DATABASE=Urdu Chatbot;UID=sa;PWD=123')
        cursor = conn.cursor()
        cursor.execute(
            "insert into ChatMessage values('"+emotion+f"','"+source+"',N'"+text+"','"+entry_date+f"',{session_id})", )
        conn.commit()
        conn.close()
        return jsonify({'message': 'Data saved successfully'}), 200
    except pyodbc.Error as e:
        print("Error occurred while saving data:", e)
        return jsonify({'error': 'Failed to save data'}), 500

@app.route('/generateReply', methods=['POST'])
def handle_generate_reply():
    path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
    doc1 = pd.read_csv(path_name, encoding='utf-8')
    doc2=doc1
    doc3 = doc2
    doc4 = doc3
    doc5 = doc4

    # Ensure all values in 'question' and 'answer' columns are strings
    doc1['question'] = doc1['question'].astype(str).fillna('')
    doc1['answer'] = doc1['answer'].astype(str).fillna('')
    doc1['Q_emotion'] = doc1['question'].astype(str).fillna('')
    doc1['A_emotion'] = doc1['answer'].astype(str).fillna('')
    data = request.json
    sentence = data.get('sentence')
    chatbot_type = data.get('type')

    if not sentence or not chatbot_type:
        return jsonify({'error': 'No sentence or chatbot type provided'}), 400

    # Assuming `predict_emotion` is defined elsewhere to predict emotion based on `sentence`
    emotion = predict_emotion(sentence)
    reply="Orignal Reply and emotion detected :"+emotion
    AllSentences=" "

    if chatbot_type.lower() == 'male':
        #reply = find_best_answer(sentence, male_model, male_vectorizer, qa_data_male)
        reply = find_best_answers(sentence, doc1)
        reply=reply+female_to_male_2nd_person(reply)

        if(emotion=="Neutral"):
            reply = reply + replyWithEmotion("Happy", sentence)

            reply = reply + replyWithEmotion("Sad", sentence)

            reply = reply + replyWithEmotion("Angry", sentence)

            reply = reply + replyWithEmotion("Fear", sentence)
        elif(emotion=="Happy"):
            reply = reply + replyWithEmotion("Sad", sentence)

            reply = reply + replyWithEmotion("Angry", sentence)

            reply = reply + replyWithEmotion("Fear", sentence)
            reply = reply + replyWithEmotion("Neutral", sentence)
        elif (emotion == "Sad"):
            reply = reply + replyWithEmotion("Happy", sentence)

            reply = reply + replyWithEmotion("Angry", sentence)

            reply = reply + replyWithEmotion("Fear", sentence)
            reply = reply + replyWithEmotion("Neutral", sentence)
        elif (emotion == "Angry"):
            reply = reply + replyWithEmotion("Sad", sentence)

            reply = reply + replyWithEmotion("Happy", sentence)

            reply = reply + replyWithEmotion("Fear", sentence)
            reply = reply + replyWithEmotion("Neutral", sentence)
        elif (emotion == "Fear"):
            reply = reply + replyWithEmotion("Sad", sentence)

            reply = reply + replyWithEmotion("Happy", sentence)

            reply = reply + replyWithEmotion("Angry", sentence)
            reply = reply + replyWithEmotion("Neutral", sentence)



    elif chatbot_type.lower() == 'angry':
        reply = find_best_answer(sentence, angry_model, angry_vectorizer, qa_data_angry)
    elif chatbot_type.lower() == 'female':
        # Assuming `male_to_female_1st_person` is defined to convert male reply to female first person
        male_reply=find_best_answer(sentence, male_model, male_vectorizer, qa_data_male)
        #male_reply = find_best_answers(sentence, doc)
        reply = male_to_female_1nd_person(male_reply)
        if (emotion == "Neutral"):
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Happy", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Sad", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Angry", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Fear", sentence))
        elif (emotion == "Happy"):
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Sad", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Angry", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Fear", sentence))
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Neutral", sentence))
        elif (emotion == "Sad"):
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Happy", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Angry", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Fear", sentence))
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Neutral", sentence))
        elif (emotion == "Angry"):
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Sad", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Happy", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Fear", sentence))
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Neutral", sentence))
        elif (emotion == "Fear"):
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Sad", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Happy", sentence))

            reply = reply + male_to_female_1nd_person(replyWithEmotion("Angry", sentence))
            reply = reply + male_to_female_1nd_person(replyWithEmotion("Neutral", sentence))
    else:
        return jsonify({'error': 'Invalid chatbot type provided'}), 400
    pos_tags = pos_tagging(sentence)
    reply=reply+"....Parts of speeches:..."
    for word, tag in pos_tags:
        reply=reply+f"{word}: {tag}"
        reply=reply+"   "
        print(f"{word}: {tag}")
    if(noun_Gender(sentence)==""):
        noun_g="Not found"
    else:
        noun_g=noun_Gender(sentence)
    reply=reply+"....Gender of nowns:...."+noun_g
    reply=reply+AllSentences





    return jsonify({'reply': reply, 'emotion': emotion}), 200

def replyWithEmotion(emotion,sentencess):
    doc1=doc[doc['Q_emotion'] == emotion]
    print(doc1.head())
    Sentenc="......."
    Sentenc = Sentenc + emotion+"=>"
    Sentenc = Sentenc + find_best_answers(sentencess, doc1)
    return Sentenc

@app.route('/create_session', methods=['POST'])
def create_session():
    data = request.json
    title = data.get('title')
    user_id = data.get('userId')
    bot_id = data.get('bot_id')

    if not title or not user_id or not bot_id:
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=HASSAN;DATABASE=Urdu Chatbot;UID=sa;PWD=123')
        cursor = conn.cursor()
        query = (
            "INSERT INTO ChatSession (Title, creationDate, userId, bot_id) "
            "VALUES (N'" + title + "', '" + str(datetime.datetime.now()) + "', " + str(user_id) + ", " + str(bot_id) + ")"
        )
        cursor.execute(query)
        conn.commit()
        new_session_id = cursor.execute("SELECT @@IDENTITY AS new_session_id").fetchone().new_session_id
        conn.close()
        return jsonify({'message': 'Session created successfully', 'sessionId': new_session_id}), 200
    except pyodbc.Error as e:
        print("Error occurred while creating session:", e)
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/user_id/<username>', methods=['GET'])
def get_user_id(username):
    session = Session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if user:
        return jsonify({'user_id': user.userId}), 200
    else:
        return jsonify({'error': 'User not found'}), 404

@app.route('/get_emotion', methods=['POST'])
def get_emotion():
    data = request.json
    sentence = data.get('sentence')

    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    emotion = predict_emotion(sentence)
    return jsonify({'emotion': emotion}), 200

@app.route('/delete_session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        session = Session()
        session.query(ChatMessage).filter_by(sessionId=session_id).delete()
        session_obj = session.query(ChatSession).filter_by(SessionId=session_id).first()

        if not session_obj:
            session.close()
            return jsonify({'error': 'Session not found'}), 404

        session.delete(session_obj)
        session.commit()
        session.close()
        return jsonify({'message': 'Session deleted successfully'}), 200
    except Exception as e:
        print("Error occurred while deleting session:", e)
        return jsonify({'error': 'Failed to delete session'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
