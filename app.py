from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from asr import transcribe_audio
from topic_extraction import extract_topics, analyze_sentiments
from insights import generate_insights
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process file (assuming it's audio or video)
        transcription = transcribe_audio(file_path)
        topics = extract_topics(transcription)
        sentiments = analyze_sentiments(transcription)
        insights = generate_insights(transcription)

        return render_template('result.html', transcription=transcription, topics=topics, sentiments=sentiments, insights=insights)

if __name__ == '__main__':
    app.run(debug=True)
