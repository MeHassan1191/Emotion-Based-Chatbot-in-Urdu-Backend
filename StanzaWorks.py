import stanza

# Download the Urdu model if you haven't already
stanza.download('ur')

# Initialize the pipeline for Urdu
nlp = stanza.Pipeline('ur')


# Function to perform POS tagging
def pos_tagging(sentence):
    # Process the sentence
    doc = nlp(sentence)

    # Extract words and their POS tags
    pos_tags = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
    print(pos_tags)

    return pos_tags


def male_to_female_1nd_person(sentence):
    # Process the sentence with Stanza
    doc = nlp(sentence)

    # Initialize a list to store modified words
    modified_words = []

    # Iterate through each word in the sentence
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['AUX', 'VERB'] and word.text.endswith('ا'):
                modified_word = word.text[:-1] + 'ی'
            else:
                modified_word = word.text
            modified_words.append(modified_word)

    # Join the modified words back into a sentence
    modified_sentence = ' '.join(modified_words)

    return modified_sentence


def female_to_male_1nd_person(sentence):
    # Process the sentence with Stanza
    doc = nlp(sentence)

    # Initialize a list to store modified words
    modified_words = []

    # Iterate through each word in the sentence
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['AUX', 'VERB'] and word.text.endswith('ی'):
                modified_word = word.text[:-1] + 'ا'
            else:
                modified_word = word.text
            modified_words.append(modified_word)

    # Join the modified words back into a sentence
    modified_sentence = ' '.join(modified_words)

    return modified_sentence

def male_to_female_2nd_person(sentence):
    # Process the sentence with Stanza
    doc = nlp(sentence)

    # Initialize a list to store modified words
    modified_words = []

    # Iterate through each word in the sentence
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['AUX', 'VERB','PRON'] and word.text.endswith('ے'):
                modified_word = word.text[:-1] + 'ی'
            else:
                modified_word = word.text
            modified_words.append(modified_word)

    # Join the modified words back into a sentence
    modified_sentence = ' '.join(modified_words)

    return modified_sentence


def female_to_male_2nd_person(sentence):
    # Process the sentence with Stanza
    doc = nlp(sentence)

    # Initialize a list to store modified words
    modified_words = []

    # Iterate through each word in the sentence
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['AUX', 'VERB','PRON'] and word.text.endswith('ی'):
                modified_word = word.text[:-1] + 'ے'
            else:
                modified_word = word.text
            modified_words.append(modified_word)

    # Join the modified words back into a sentence
    modified_sentence = ' '.join(modified_words)

    return modified_sentence

# Example usage
sentence = "ہیلو ، آج آپ کیسی ہیں ؟"
pos_tags = pos_tagging(sentence)
print(female_to_male_2nd_person(sentence))

# Print the POS tags
for word, tag in pos_tags:
    print(f"{word}: {tag}")

def noun_Gender(sentence):
    pos_tags = pos_tagging(sentence)
    gender_Dic = {
        "آج":"male",
        "جگہ": "female",
        "سال": "male",
        "بارش": "female",
        "اجزاء": "male",
        "موسم": "male",
        "شام": "female",
        "خوشی": "female",
        "فلم": "female",
        "وقت": "male",
        "جانور": "male",
        "بھائی": "male",
        "منصوبہ": "male",
        "ضرورت": "female",
        "خیال": "male",
        "دفتر": "male",
        "تھکاوٹ": "female",
        "گرمی": "female",
        "محبت": "female",
        "چائے": "female",
        "بازار": "male",
        "کتاب": "female",
        "گھر": "male",
        "اسکول": "male",
        "دھوکہ": "male",
        "بچے": "male",
        "نیند": "female",
        "چیز": "female",
        "دن": "male",
        "خوشی": "female",

    }
    sent=""
    # Print the POS tags
    for word, tag in pos_tags:
        print(f"{word}: {tag}")
        if(tag=="NOUN" and word in gender_Dic):
            sent=sent+ word+":"+gender_Dic[word]
    return sent










