import re
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the model
loaded_model = load_model('question_answer_model.keras')

# Load questions and answers from the JSON file
with open('qa_data.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

questions = qa_data['questions']
answers = qa_data['answers']

# Vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions).toarray()

unique_answers = np.array(answers)

def find_best_answer(input_text, model, vectorizer):
    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text]).toarray()
    prediction = model.predict(input_vector)
    best_answer_index = np.argmax(prediction)
    best_answer = unique_answers[best_answer_index]
    return best_answer

# Simulate 50 common conversational Urdu sentences
urdu_sentences = [
    "مجھے آپ سے مل کر خوشی ہوئی",
    "آج کا دن بہت اچھا ہے",
    "مجھے بھوک لگی ہے",
    "مجھے نیند آ رہی ہے",
    "میں ٹی وی دیکھ رہا ہوں",
    "میں کتاب پڑھ رہا ہوں",
    "آج کا کام ختم ہو گیا ہے",
    "میں ابھی بازار جا رہا ہوں",
    "مجھے چائے پینی ہے",
    "آج بہت گرمی ہے",
    "بہت تھکاوٹ محسوس ہو رہی ہے",
    "مجھے ورزش کرنی ہے",
    "میں دفتر میں ہوں",
    "آج بہت مصروف دن تھا",
    "کل ملاقات ہوگی",
    "آپ کیسے ہیں؟",
    "آپ کا دن کیسا گزر رہا ہے؟",
    "مجھے فلمیں دیکھنا پسند ہے",
    "آج شام کو ہم ملیں گے",
    "آپ کہاں جا رہے ہیں؟",
    "کیا آپ نے کھانا کھایا؟",
    "مجھے تھوڑا آرام کرنا ہے",
    "آج بارش ہو رہی ہے",
    "میں سفر پر جا رہا ہوں",
    "میں نے نیا فون خریدا ہے",
    "مجھے یہ جگہ بہت پسند ہے",
    "آپ کا پسندیدہ رنگ کون سا ہے؟",
    "مجھے موسم سرما پسند ہے",
    "آج بہت سردی ہے",
    "میری طبیعت ٹھیک نہیں ہے",
    "مجھے ڈاکٹر کے پاس جانا ہے",
    "آپ کے بچوں کے نام کیا ہیں؟",
    "آپ کا گھر بہت خوبصورت ہے",
    "مجھے اردو بولنا بہت پسند ہے",
    "میں دوستوں کے ساتھ باہر جا رہا ہوں",
    "آپ کا خاندان کیسا ہے؟",
    "مجھے موسیقی سننا پسند ہے",
    "آج کا کھانا بہت مزیدار تھا",
    "مجھے تازہ ہوا میں چلنا ہے",
    "آپ کی طبیعت کیسی ہے؟",
    "آپ کیا کر رہے ہیں؟",
    "مجھے آپ کی مدد چاہیے",
    "کیا آپ میرے ساتھ چلیں گے؟",
    "آپ کا پسندیدہ کھانا کیا ہے؟",
    "مجھے فلمیں دیکھنا بہت پسند ہے",
    "آپ کا دن کیسا گزر رہا ہے؟",
    "آپ کے کتنے بہن بھائی ہیں؟",
    "مجھے آپ کا جواب چاہیے",
    "آپ کو یہ پسند آیا؟"
]

# Process each sentence and print the predicted answer
for sentence in urdu_sentences:
    best_answer = find_best_answer(sentence, loaded_model, vectorizer)
    print(f"Sentence: {sentence}")
    print(f"Predicted answer: {best_answer}\n")


#
# #             Loading Angry Model
# import re
# import numpy as np
# import json
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     return text
#
# # Load the model
# loaded_model = load_model('question_answer_model_angry.keras')
# loaded_model.summary()
#
# # Load questions and answers from the JSON file
# with open('qa_data_angry.json', 'r', encoding='utf-8') as f:
#     qa_data = json.load(f)
#
# questions = qa_data['questions']
# answers = qa_data['answers']
#
# # Vectorize the questions
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(questions).toarray()
#
# unique_answers = np.array(answers)
#
# def find_best_answer(input_text, model, vectorizer):
#     input_text = preprocess_text(input_text)
#     input_vector = vectorizer.transform([input_text]).toarray()
#     prediction = model.predict(input_vector)
#     best_answer_index = np.argmax(prediction)
#     best_answer = unique_answers[best_answer_index]
#     return best_answer
#
# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter your question (type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             break
#
#         best_answer = find_best_answer(user_input, loaded_model, vectorizer)
#
#         if best_answer:
#             print("Best matching answer:", best_answer)
#         else:
#             print("No suitable answer found.")
