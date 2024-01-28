from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text
pdf_path = "48lawsofpower.pdf"

pdf_text = extract_text_from_pdf(pdf_path)
passages = pdf_text

nltk.download('punkt')
nltk.download('stopwords')

questions = ["Can you give me an example from history where the enemy was crushed totally?"] 

stop_words = set(stopwords.words('english'))
preprocessed_passages = [' '.join([word.lower() for word in nltk.word_tokenize(p) if word.isalnum() and word.lower() not in stop_words]) for p in passages]

preprocessed_questions = [' '.join([word.lower() for word in nltk.word_tokenize(q) if word.isalnum() and word.lower() not in stop_words]) for q in questions]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_passages + preprocessed_questions)

similarity_scores = (tfidf_matrix[:len(passages)] @ tfidf_matrix[len(passages):].T).toarray()

for i, question in enumerate(questions):
    question_index = len(passages) + i
    top_passage_index = similarity_scores[i].argmax()

    answer = passages[top_passage_index]
    print(f"Question: {question}\nAnswer: {answer}\n")

