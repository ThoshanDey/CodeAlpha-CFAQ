import json
import re
import nltk
import tkinter as tk
from tkinter import scrolledtext
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

with open("faq_data.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

stop_words = set(stopwords.words("english"))

import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_response(user_input):
    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    print("User:", user_input)
    print("Processed:", processed_input)
    print("Best Score:", best_score)
    print("Matched Question:", questions[best_match_index])

    if best_score < 0.05:
      return "Sorry, I couldn't find a relevant answer."

    return answers[best_match_index]

def send_message():
    user_text = user_entry.get()

    if not user_text.strip():
        return

    chat_area.insert(tk.END, f"You: {user_text}\n")

    bot_response = get_response(user_text)

    chat_area.insert(
        tk.END,
        f"Bot: {bot_response}\n\n"
    )

    user_entry.delete(0, tk.END)

root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("700x500")

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=80,
    height=25
)

chat_area.pack(padx=10, pady=10)

user_entry = tk.Entry(root, width=60)
user_entry.pack(side=tk.LEFT, padx=10, pady=10)

send_button = tk.Button(
    root,
    text="Send",
    command=send_message
)

send_button.pack(side=tk.LEFT)

root.mainloop()