import re
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import Binary
import numpy as np
import urllib.parse

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model and NLTK stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

from urllib.parse import quote_plus

# Your MongoDB Atlas credentials
username = "admin"
password = "Test@123"

escaped_username = urllib.parse.quote_plus(username)
escaped_password = urllib.parse.quote_plus(password)

# Create the MongoDB connection string
connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@cluster0.mszvmdv.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(connection_string, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["job_database"]
collection = db["job_collection"]

def preprocess_job_description(description_text):
    """
    Preprocesses a job description text for vectorization.

    Args:
        description_text: The text of the job description.

    Returns:
        A dictionary containing the extracted features and generated vectors.
    """
    # Lowercase and remove diacritics
    description_text = description_text.lower()
    description_text = re.sub(r"[^\w\s]", "", description_text)

    # Tokenize and remove stopwords
    tokens = [token for token in nltk.word_tokenize(description_text) if token not in stop_words]

    # Compute spaCy-based job description vector
    doc = nlp(description_text)
    description_vector = doc.vector

    # Save extracted features and vectors
    preprocessed_data = {
        "tokens": tokens,
        "description_vector": description_vector
    }
    return preprocessed_data

def extract_job_fields(description_text):
    # Initialize additional fields with default values
    additional_fields = {
        "id": None,
        "link": None,
        "title": None,
        "location": None,
        "sponsorship": None,
        "skillset": None,
        "description_vector": None
    }

    # Extract job ID (if present)
    id_match = re.search(r"\bID: (\d+)", description_text)
    if id_match:
        additional_fields["id"] = int(id_match.group(1))

    # Extract job link (if present)
    link_match = re.search(r"https?://[^\s]+", description_text)
    if link_match:
        additional_fields["link"] = link_match.group(0)

    # Extract job title (if present)
    title_match = re.search(r"\bTitle: (.+)", description_text)
    if title_match:
        additional_fields["title"] = title_match.group(1)

    # Extract job location (if present)
    location_match = re.search(r"\bLocation: (.+)", description_text)
    if location_match:
        additional_fields["location"] = location_match.group(1)

    # Extract sponsorship information (if present)
    sponsorship_match = re.search(r"\bSponsorship: (.+)", description_text)
    if sponsorship_match:
        additional_fields["sponsorship"] = sponsorship_match.group(1)

    # Extract skillset (if present)
    skillset_match = re.search(r"\bSkillset: (.+)", description_text)
    if skillset_match:
        additional_fields["skillset"] = [skill.strip() for skill in skillset_match.group(1).split(",")]

    return additional_fields

def insert_into_mongodb(data):
    try:
        # Check if 'description_vector' key is present
        if 'description_vector' in data:
            # Convert 'description_vector' to a list if it's a numpy array
            if isinstance(data['description_vector'], np.ndarray):
                data['description_vector'] = data['description_vector'].tolist()

            # Insert the data into MongoDB
            collection.insert_one(data)
            print("Data inserted successfully.")
        else:
            print("Key 'description_vector' not found in data.")

    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")

# Example usage
job_description_text = """
BTS' Jungkook likely held his final live session before entering mandatory military service. The K-pop sensation, along with fellow band members RM, Jimin, and Taehyung, is preparing for an 18-month enlistment. The Standing Next to You singer engaged with fans on Weverse live, reminiscing about cherished moments and sharing heartfelt stories with the fandom. The session took a hilarious turn when the eldest member, Jin, crashed the session and also offered Jungkook some advice ahead of his impending enlistment.

Jungkook and Jin(Fan Cam X pic)
Jungkook and Jin(Fan Cam X pic)
Also read: BTS is unbeatable: GTA 6 trailer racks up millions but fails to dethrone Dynamite views; fans demand collaboration

Stay tuned with breaking news on HT Channel on Facebook. Join Now
BTS’ Jin offers words of encouragement to Jungkook during live stream
Following the recent live session with all remaining BTS members, Jungkook casually connected with his fans on his Weverse account, engaging in conversation as he typically does since he chose to deactivate his Instagram. Amidst his heartfelt farewell, Jin made a surprise entrance with his iconic "Ke Ke Ke Ke Ke" move, playfully teasing the golden maknae. The Moon singer's infectious laughter throughout the session added a lighthearted touch, bringing joy to fans who were initially feeling overwhelmed by Jungkook's absence in the coming days.

Jin also shared a piece of advice to the golden maknae. The Epiphany singer who is now part of the "Elite class of warriors" in the military asked the youngest to remember and "memorize the ROK Army’s gymnastics during the exercise time.”

Shortly afterward, the comment section was flooded with ARMYs expressing their emotions through teary eyes. A fan wrote “JUNGKOOK LIVE AND SEOKJIN IS LAUGHING, PLEASE”, while others said “Jungkooka take these advice seriously”, “Jin showing his hyung power”, “That’s some advice from the senior”, “Jungkook will love doing exercise with his unit”, “Omo he will definitely show off his skills during the exercise”.
"""

# Extract additional fields
additional_fields = extract_job_fields(job_description_text)

# Preprocess job data
preprocessed_job_data = preprocess_job_description(job_description_text)

# Combine additional fields and preprocessed data
job_data = {**additional_fields, **preprocessed_job_data}

# Print extracted fields
print("Additional Fields:", additional_fields)

# Insert data into MongoDB
insert_into_mongodb(job_data)