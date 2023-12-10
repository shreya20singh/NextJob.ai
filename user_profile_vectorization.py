from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from pymongo import MongoClient
from bson import Binary, ObjectId
from sentence_transformers import SentenceTransformer
import re
import spacy
import nltk
from nltk.corpus import stopwords
import numpy as np
import urllib.parse

app = Flask(__name__)
api = Api(app)

# Your MongoDB Atlas credentials
username = "admin"
password = "Test@123"

# Ensure that you've downloaded the spaCy model and NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

escaped_username = urllib.parse.quote_plus(username)
escaped_password = urllib.parse.quote_plus(password)

# Create the MongoDB connection string
connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.mszvmdv.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(connection_string)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Access the job_database and job_collection
db = client["job_database"]
collection = db["job_collection"]

# Define stopwords
stop_words = set(stopwords.words("english"))


def preprocess_resume(resume_text):
    # Lowercase and remove diacritics

    # Extract Title, Work Experience, and Location
    title_match = re.search(r'Title:(.+?)(?:\n|$)', resume_text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else None

    experience_match = re.search(r'Work Experience:(.+?)(?:\n|$)', resume_text, re.IGNORECASE)
    work_experience = experience_match.group(1).strip() if experience_match else None

    location_match = re.search(r'Location:(.+?)(?:\n|$)', resume_text, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else None

    resume_text = resume_text.lower()
    resume_text = re.sub(r"[^\w\s]", "", resume_text)

    # Tokenize and remove stopwords
    tokens = [token for token in nltk.word_tokenize(resume_text) if token not in stop_words]

    # Extract named entities (optional)
    named_entities = spacy.load("en_core_web_sm")(resume_text).ents

    # Generate TF-IDF vector
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform([resume_text])
    feature_names = vectorizer.get_feature_names_out()

    # Compute spaCy-based resume vector
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    resume_vector = doc.vector

    # Update preprocessed_data
    preprocessed_data = {
        "tokens": tokens,
        "features": features,
        "feature_names": feature_names,
        "named_entities": named_entities,
        "resume_vector": resume_vector,
        "title": title,
        "work_experience": work_experience,
        "location": location
    }
    return preprocessed_data


class ResumeSearch(Resource):
    def post(self):
        data = request.get_json()

        # Extract resume text from the request data
        resume_text = data.get('resume_text', '')

        # Preprocess resume
        preprocessed_data = preprocess_resume(resume_text)

        # Extract values for the vector_search_query
        title = preprocessed_data["title"]
        work_experience = preprocessed_data["work_experience"]
        location = preprocessed_data["location"]

        similarity_threshold = 1.5

        resume_vector_list = preprocessed_data["resume_vector"].tolist()
        normalized_query_vector = normalize([resume_vector_list])[0].tolist()

        vector_search_query = [
            {
                "$vectorSearch": {
                    "index": "similarity_search",
                    "path": "description_vector",
                    "queryVector": normalized_query_vector,
                    "numCandidates": 100,
                    "limit": 10
                }
            },
            {
                '$project': {
                    '_id': 1,
                    'plot': 1,
                    'title': 1,
                    'link': 1,
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            }
        ]

        results = list(collection.aggregate(vector_search_query))

        # Convert ObjectId to string for serialization
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])

        # Filter results based on the similarity threshold
        filtered_results = [result for result in results if "score" in result and result["score"] > similarity_threshold]

        return jsonify(filtered_results)


# Add the following lines outside the ResumeSearch class
api.add_resource(ResumeSearch, '/search')

if __name__ == '__main__':
    app.run(debug=True)
