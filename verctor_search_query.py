import re
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from pymongo import MongoClient
import numpy as np
import urllib.parse
from urllib.parse import quote_plus

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

# Example usage
resume_text = """
Shreya 
Title: iOS Engineer
Work Experience: 3
Location: United States

SUDHANSHU PANDEY
New York, NY, US  sp6370@nyu.edu  (929) 854-2845   https://www.linkedin.com/in/sp6370/  
EDUCATION
Master of Science in Computer Science
New York University • New York, NY • 2023
Master and Bachelor of Technology in Software Engineering
Vellore Institute of Technology • Vellore, India • 2021
EXPERIENCE
Software Engineer 
Nodal Health	July 2023 - October 2023, New York, NY
• Boosted system reliability by reengineering webhook event handling to mitigate race conditions, ensuring accurate user status updates and seamless progression through verification stages for over 20% of impacted users.
• Enhanced Django Rest API testing with Pytest, achieving robust permission validation for multiple roles and a 30% surge in test coverage, bolstering product stability and user confidence.
• Improved the developer experience by authoring extensive GitHub README documentation that clearly detailed CI/CD processes for the backend, with a specific focus on seamlessly integrating Docker and GitHub Actions.
Course Assistant
New York University	February 2023 - May 2023, New York, NY
• Engaged with a class of 200 students as a course assistant for Operating Systems under Professor Gustavo at NYU, offering 150+ minutes of weekly office hours to provide comprehensive guidance on OS concepts, C, and Assembly in both individual and group settings.
Software Engineer Intern
Nodal Health	June 2022 - December 2022, New York, NY
• Utilized Django to seamlessly incorporate the HubSpot API into the backend, leading to a 35% enhancement in the marketing team's efficiency by enabling automated status tracking.
• Drastically improved resource efficiency by optimizing event collection through Sentry Python SDK, resulting in a 90% reduction in collected transactions and maximizing the utilization of limited Sentry quota. 
• Developed a feature on the company's Django Staff website using PDFkit that exports user health information as PDF.
Software Engineer Intern
MyCarmunity	April 2021 - July 2021, Halle, Germany
• Developed a server-side payment processing solution using the Paypal JavaScript SDK, reducing average transaction completion time by 35% through an improved understanding of the PayPal API and writing unit tests for the back end.
• Implemented a scalable microservice for sending 10,000 personalized emails/day using Firebase Cloud Functions, Node.js and SendGrid, resulting in improved customer satisfaction and a 10% increase in overall email open rates.
• Built interactive UI prototypes using React to demonstrate new features and enhancements for the marketing website. Collaborated with designers to iterate based on user feedback. 
• Wrote efficient and well-structured code for making error-free requests, queries and validations to Firestore through Node.js.
PROJECTS
The Bird
Distributed Systems Course @ NYU • September 2022 - December 2022
• Implemented a Golang RESTful API with JWT authentication to connect a React.js frontend for a Twitter clone application.
• Optimized inter-service communication by developing a stateless gRPC wrapper server, achieving a 10% reduction in latency within the microservices architecture.
Smarty Pictures
Cloud Computing and Big Data Systems Course @ NYU • September 2021 - December 2021
• Built serverless microservice driven web app to search photos by text and voice using React, Node.js, ElasticSearch and AWS.
• Developed a CI/CD pipeline using AWS Code Build, Code Pipeline, and CloudFormation for automated deployment to AWS from GitHub.
SKILLS
Languages: Python, JavaScript, TypeScript, C++, Go, SQL, HTML, CSS
Frameworks: Django, Django Rest, React.js, Flask, Node.js, Express
Databases and Tools: PostgreSQL, Firebase, MongoDB, Docker, Git, Jira, Selenium, Postman, Sentry, Datadog
AWS: Lambda, EC2, S3, DynamoDB, CloudWatch, API Gateway, SQS, SES

"""

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

i = 0
for result in results:
    if "score" in result and result["score"] > similarity_threshold:
        # Update the results array only for scores greater than 80%
        i += 1
        print("Object ID:", result.get("_id", "No _id field"))
        print(i)
        print(result)
        print("\n")
    else:
        i += 1
        print(i)
        print(result["title"])
        print(result["score"])
        print("No _id field in the result or score is below the threshold.")

# Close the connection
client.close()