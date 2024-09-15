import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

# Load spaCy and Sentiment model
nlp = spacy.load('en_core_web_sm')
sentiment_analysis = pipeline("sentiment-analysis")

def extract_topics(text, num_topics=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_vectorized = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_vectorized)
    
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = [[feature_names[i] for i in topic.argsort()[-5:]] for topic in topics]
    
    return topic_keywords

def analyze_sentiments(text):
    doc = nlp(text)
    sentiments = []
    for sent in doc.sents:
        result = sentiment_analysis(sent.text)
        sentiments.append({"sentence": sent.text, "sentiment": result[0]['label'], "score": result[0]['score']})
    return sentiments
