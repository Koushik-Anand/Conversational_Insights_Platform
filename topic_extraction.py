import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

# Load spaCy and Sentiment model
nlp = spacy.load('en_core_web_sm')
sentiment_analysis = pipeline("sentiment-analysis")

def extract_topics(texts, num_topics=3):
    """
    Extracts topics from a list of documents using LDA.
    
    Args:
        texts (list of str): List of input text documents.
        num_topics (int): The number of topics to extract.
    
    Returns:
        topic_keywords (list of list): Keywords representing each topic.
    """
    if isinstance(texts, str):  # If a single string is passed, convert it to a list
        texts = [texts]
    
    vectorizer = CountVectorizer(max_df=1.0, min_df=1)
    text_vectorized = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_vectorized)
    
    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the top 5 keywords for each topic
    topic_keywords = [[feature_names[i] for i in topic.argsort()[-5:]] for topic in topics]
    
    return topic_keywords

def analyze_sentiments(text):
    """
    Analyzes the sentiment of the input text.
    
    Args:
        text (str): The input text for sentiment analysis.
    
    Returns:
        sentiments (list of dict): List of sentences with their sentiment labels and scores.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]  # Extract sentences
    sentiment_results = sentiment_analysis(sentences)  # Use batch processing for efficiency
    
    sentiments = [
        {
            "sentence": sent,
            "sentiment": result['label'],
            "score": result['score']
        }
        for sent, result in zip(sentences, sentiment_results)
    ]
    
    return sentiments
