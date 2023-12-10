import re
import spacy
import nltk
from nltk.corpus import stopwords
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, OnSiteOrRemoteFilters
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model and NLTK stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

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

def extract_job_fields(description_text):
    # Initialize additional fields with default values
    additional_fields = {
        "id": None,
        "location": None,
        "sponsorship": None,
        "skillset": None,
        "description_vector": None
    }

    # Extract job ID (if present)
    id_match = re.search(r"\bID: (\d+)", description_text)
    if id_match:
        additional_fields["id"] = int(id_match.group(1))

    

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

    # Check if description vector is already present
    if additional_fields["description_vector"] is None:
        # Compute spaCy-based job description vector
        doc = nlp(description_text)
        additional_fields["description_vector"] = doc.vector.tolist()

    return additional_fields


# Fired once for each successfully processed job
# Fired once for each successfully processed job
def on_data(data: EventData):
    try:
        # Print specific fields from the 'data' object
        print('[ON_DATA] Title:', data.title)
        print('[ON_DATA] Company:', data.company)
        print('[ON_DATA] Company Link:', data.company_link)
        print('[ON_DATA] Date:', data.date)
        print('[ON_DATA] Link:', data.link)
        print('[ON_DATA] Insights:', data.insights)
        print('[ON_DATA] Description Length:', len(data.description))

        # Preprocess job data
        preprocessed_job_data = preprocess_job_description(data.description)

        # Extract additional fields
        additional_fields = extract_job_fields(data.description)

        # Combine additional fields and preprocessed data
        job_data = {
            "title": data.title,
            "company": data.company,
            "company_link": data.company_link,
            "date": data.date,
            "link": data.link,
            "insights": data.insights,
            "description_length": len(data.description),
            **additional_fields,
            **preprocessed_job_data
        }

        # Insert data into MongoDB
        insert_into_mongodb(job_data)

    except Exception as e:
        print(f"Error processing data: {e}")



# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
    print('[ON_METRICS]', str(metrics))

def on_error(error):
    print('[ON_ERROR]', error)

def on_end():
    print('[ON_END]')

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

scraper = LinkedinScraper(
    chrome_executable_path=None,  # Custom Chrome executable path (e.g., /foo/bar/bin/chromedriver) 
    chrome_options=None,  # Custom Chrome options here
    headless=True,  # Overrides headless mode only if chrome_options is None
    max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
    slow_mo=0.5,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
    page_load_timeout=40  # Page load timeout (in seconds)    
)

# Add event listeners
scraper.on(Events.DATA, on_data)
scraper.on(Events.ERROR, on_error)
scraper.on(Events.END, on_end)

queries = [
    Query(
        options=QueryOptions(
            limit=500  # Limit the number of jobs to scrape.            
        )
    ),
    Query(
        query='Engineer',
        options=QueryOptions(
            locations=['United States'],
            apply_link=True,  # Try to extract apply link (easy applies are skipped). If set to True, scraping is slower because an additional page must be navigated. Default to False.
            skip_promoted_jobs=True,  # Skip promoted jobs. Default to False.
            page_offset=2,  # How many pages to skip
            limit=5,
            filters=QueryFilters(
                company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies.                
                relevance=RelevanceFilters.RECENT,
                time=TimeFilters.MONTH,
                type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE],
                experience=[ExperienceLevelFilters.MID_SENIOR]
            )
        )
    ),
]

scraper.run(queries)
