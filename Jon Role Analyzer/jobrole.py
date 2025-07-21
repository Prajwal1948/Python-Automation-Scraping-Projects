import time
import pandas as pd
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load English NLP model
nlp = spacy.load("en_core_web_sm")  # Don't import it separately

# Step 1: Scrape job data from Indeed
driver = webdriver.Chrome()
driver.get("https://www.indeed.com/jobs?q=python+developer&limit=50")
time.sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")
jobs = soup.find_all("div", class_="job_seen_beacon")

job_list = []
for job in jobs:
    title = job.find("h2").text.strip() if job.find("h2") else "N/A"
    company = job.find("span", class_="companyName").text.strip() if job.find("span", class_="companyName") else "N/A"
    location = job.find("div", class_="companyLocation").text.strip() if job.find("div",
                                                                                  class_="companyLocation") else "N/A"
    desc = job.find("div", class_="job-snippet").text.strip().replace("\n", " ") if job.find("div",
                                                                                             class_="job-snippet") else "N/A"
    job_list.append([title, company, location, desc])

driver.quit()

# Step 2: Create DataFrame
df = pd.DataFrame(job_list, columns=["Title", "Company", "Location", "Description"])


# Step 3: Clean and lemmatize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])


df['cleaned_desc'] = df['Description'].apply(clean_text)
df['lemmas'] = df['cleaned_desc'].apply(lemmatize_text)

# Step 4: Extract skills
target_skills = ['python', 'django', 'flask', 'sql', 'aws', 'docker', 'react', 'node', 'pandas', 'ml']


def extract_skills(text):
    return ', '.join([skill for skill in target_skills if skill in text])


df['skills_found'] = df['lemmas'].apply(extract_skills)

# Step 5: Cluster job roles
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['lemmas'])

kmeans = KMeans(n_clusters=3, random_state=42)
df['job_cluster'] = kmeans.fit_predict(X)

# Step 6: Optional - Match with resume
resume_text = "python sql aws docker pandas flask"
resume_vec = vectorizer.transform([resume_text])
df['match_score'] = cosine_similarity(X, resume_vec).flatten()

# Step 7: Save and print output
df_sorted = df.sort_values(by='match_score', ascending=False)
df_sorted.to_csv("job_role_analysis.csv", index=False)

print("\nTop Matching Jobs:")
print(df_sorted[['Title', 'Company', 'skills_found', 'match_score']].head())
