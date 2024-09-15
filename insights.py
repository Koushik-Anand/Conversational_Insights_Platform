from transformers import pipeline

# Summarization model
summarizer = pipeline("summarization")

def generate_insights(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    # Additional custom logic to identify key takeaways or consensus.
    return summary
