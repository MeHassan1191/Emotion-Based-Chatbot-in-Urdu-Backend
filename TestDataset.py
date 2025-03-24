# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import re
# from googletrans import Translator
#
# def remove_repeated_sentences(text):
#     # Split the text into sentences using a regex that captures punctuation
#     sentences = re.split(r'(?<=[.!?]) +', text)
#
#     # Use a set to track unique sentences
#     seen_sentences = set()
#     unique_sentences = []
#
#     for sentence in sentences:
#         if sentence not in seen_sentences:
#             unique_sentences.append(sentence)
#             seen_sentences.add(sentence)
#
#     # Join the unique sentences back into a single string
#     result = ' '.join(unique_sentences)
#     return result
#
# def main():
#     # Initialize the tokenizer and model from the pre-trained GPT-2
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     model = GPT2LMHeadModel.from_pretrained('gpt2')
#
#     # Initialize the translator
#     translator = Translator()
#
#     # Function to encode user input for the model
#     def encode_input(user_input):
#         return tokenizer.encode(user_input, return_tensors='pt')
#
#     # Function to generate a response from the model
#     def generate_response(encoded_input):
#         # Generate a response with a maximum of 50 tokens
#         max_length = encoded_input.shape[1] + 50
#         output = model.generate(
#             encoded_input,
#             max_length=max_length,
#             pad_token_id=tokenizer.eos_token_id,
#             temperature=0.7,  # Adjust to control randomness
#             top_k=50,         # Limits the number of top tokens considered for generation
#             top_p=0.9,        # Nucleus sampling: considers the smallest set of tokens with cumulative probability > top_p
#             num_return_sequences=1
#         )
#         return tokenizer.decode(output[:, encoded_input.shape[-1]:][0], skip_special_tokens=True)
#
#     # Main loop to get user input and generate model responses
#     while True:
#         # Get user input
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break
#
#         # Translate Urdu input to English
#         translated_input = translator.translate(user_input, src='ur', dest='en').text
#         print(f"Translated to English: {translated_input}")
#
#         # Encode and generate response
#         encoded_input = encode_input(translated_input)
#         response = generate_response(encoded_input)
#
#         # Remove repeated sentences
#         response = remove_repeated_sentences(response)
#
#         # Translate response back to Urdu
#         translated_response = translator.translate(response, src='en', dest='ur').text
#
#         # Print the model's response in Urdu
#         print(f"Copilot: {translated_response}")
#
# if __name__ == '__main__':
#     main()



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from StanzaWorks import pos_tagging

# Read the CSV file into a DataFrame
path_name = 'C:\\Users\\hp\\Downloads\\TrainResponse.csv'
doc = pd.read_csv(path_name, encoding='utf-8')


# Ensure all values in 'question' and 'answer' columns are strings
doc['question'] = doc['question'].astype(str).fillna('')
doc['answer'] = doc['answer'].astype(str).fillna('')

def preprocess_text(text):
    # Convert to lowercase (optional for Urdu)
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

def find_best_answers(input_text, df):
    # Preprocess input text
    input_text = preprocess_text(input_text)

    # Preprocess all questions
    df['processed_question'] = df['question'].apply(preprocess_text)

    # Combine input text with the questions for vectorization
    texts = [input_text] + df['processed_question'].tolist()

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity between the input text and all questions
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Find the indices of the most similar questions
    max_similarity = cosine_similarities.max()
    best_match_indices = [i for i, similarity in enumerate(cosine_similarities) if similarity == max_similarity]

    # Select the shortest question among the most similar questions
    if best_match_indices:
        best_match_index = min(best_match_indices, key=lambda idx: len(df.iloc[idx]['question']))
        return df.iloc[best_match_index]['answer']
    else:
        return None

# urdu_sentences = [
#     "مجھے آپ سے مل کر خوشی ہوئی",
#     "آج کا دن بہت اچھا ہے",
#     "مجھے بھوک لگی ہے",
#     "مجھے نیند آ رہی ہے",
#     "میں ٹی وی دیکھ رہا ہوں",
#     "میں کتاب پڑھ رہا ہوں",
#     "آج کا کام ختم ہو گیا ہے",
#     "میں ابھی بازار جا رہا ہوں",
#     "مجھے چائے پینی ہے",
#     "آج بہت گرمی ہے",
#     "بہت تھکاوٹ محسوس ہو رہی ہے",
#     "مجھے ورزش کرنی ہے",
#     "میں دفتر میں ہوں",
#     "آج بہت مصروف دن تھا",
#     "کل ملاقات ہوگی",
#     "آپ کیسے ہیں؟",
#     "آپ کا دن کیسا گزر رہا ہے؟",
#     "مجھے فلمیں دیکھنا پسند ہے",
#     "آج شام کو ہم ملیں گے",
#     "آپ کہاں جا رہے ہیں؟",
#     "کیا آپ نے کھانا کھایا؟",
#     "مجھے تھوڑا آرام کرنا ہے",
#     "آج بارش ہو رہی ہے",
#     "میں سفر پر جا رہا ہوں",
#     "میں نے نیا فون خریدا ہے",
#     "مجھے یہ جگہ بہت پسند ہے",
#     "آپ کا پسندیدہ رنگ کون سا ہے؟",
#     "مجھے موسم سرما پسند ہے",
#     "آج بہت سردی ہے",
#     "میری طبیعت ٹھیک نہیں ہے",
#     "مجھے ڈاکٹر کے پاس جانا ہے",
#     "آپ کے بچوں کے نام کیا ہیں؟",
#     "آپ کا گھر بہت خوبصورت ہے",
#     "مجھے اردو بولنا بہت پسند ہے",
#     "میں دوستوں کے ساتھ باہر جا رہا ہوں",
#     "آپ کا خاندان کیسا ہے؟",
#     "مجھے موسیقی سننا پسند ہے",
#     "آج کا کھانا بہت مزیدار تھا",
#     "مجھے تازہ ہوا میں چلنا ہے",
#     "آپ کی طبیعت کیسی ہے؟",
#     "آپ کیا کر رہے ہیں؟",
#     "مجھے آپ کی مدد چاہیے",
#     "کیا آپ میرے ساتھ چلیں گے؟",
#     "آپ کا پسندیدہ کھانا کیا ہے؟",
#     "مجھے فلمیں دیکھنا بہت پسند ہے",
#     "آپ کا دن کیسا گزر رہا ہے؟",
#     "آپ کے کتنے بہن بھائی ہیں؟",
#     "مجھے آپ کا جواب چاہیے",
#     "آپ کو یہ پسند آیا؟"
# ]
#
# # Process each sentence and print the predicted answer
# for sentence in urdu_sentences:
#     best_answer = find_best_answers(sentence, doc)
#     pos_tags = pos_tagging(sentence)
#
#     # Print the POS tags
#     for word, tag in pos_tags:
#         print(f"{word}: {tag}")
#     pos_tags = pos_tagging(best_answer)
#
#     # Print the POS tags
#     for word, tag in pos_tags:
#         print(f"{word}: {tag}")
#     print(f"Sentence: {sentence}")
#     print(f"Predicted answer: {best_answer}\n")
#
# if __name__ == "__main__":
#     find_best_answer("کوئی مناسب جواب نہیں ملا.",doc)
#     while True:
#         user_input = input("اپنا سوال درج کریں (خروج کے لیے 'exit' لکھیں): ")
#         if user_input.lower() == 'exit':
#             break
#
#         best_answer = find_best_answer(user_input, doc)
#
#         if best_answer:
#             print("بہترین ملتا جلتا جواب:", best_answer)
#         else:
#             print("کوئی مناسب جواب نہیں ملا.")




import re
import numpy as np
import json
#from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the model
#loaded_model = load_model('question_answer_model.keras')

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

