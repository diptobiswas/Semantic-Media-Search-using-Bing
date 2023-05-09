import openai
openai.api_key = "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"

import json
import os
import requests
from requests_html import HTMLSession
from datetime import datetime, timedelta
import time
from time import sleep
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import smtplib
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# Function to delete the files in the Reports folder
def delete_files(files_to_delete):
    for file_path in files_to_delete:
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Error: {file_path} not found")

# Function to send an email with the Reports
def send_email(report_files, recipients, subject, body):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'your_email@example.com'
    msg['To'] = ', '.join(recipients)

    msg.attach(EmailMessage(body))

    for file_path in report_files:
        with open(file_path, 'rb') as f:
            file_name = os.path.basename(file_path)
            file_data = f.read()
            attachment = MIMEApplication(file_data, _subtype="octet-stream")
            attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
            msg.attach(attachment)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login('diptobiswas0007@gmail.com', 'ikqjpvwrdvlwqgky')
        server.send_message(msg)

# Function to calculate the sentiment score using TextBlob
def calculate_sentiment_score(paragraph):
    text_blob = TextBlob(paragraph)
    sentiment = text_blob.sentiment

    # sentiment.polarity is a score ranging from -1 (negative) to 1 (positive)
    polarity = sentiment.polarity

    # Normalize the polarity score to 0-1 range and multiply by 100
    percentage_score = ((polarity + 1) / 2) * 100

    return percentage_score

# Function to calculate the relevance score
def calculate_relevance_score(paragraph, article_content):
    vectorizer = TfidfVectorizer()
    texts = [paragraph, article_content]
    tfidf_matrix = vectorizer.fit_transform(texts)
    relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Scale the cosine similarity to a 0-1 range and then multiply by 100
    percentage_score = ((relevance_score[0][0] + 1) / 2) * 100
    return percentage_score

# Function to get date one week filtered URL for the Kitchener Newsroom
def construct_url(date: str, base_url: str = "https://www.kitchener.ca/Modules/News/en") -> str:
    # Parse the date input
    date_obj = datetime.strptime(date, "%m/%d/%Y")

    # Calculate the date 7 days before
    date_from_obj = date_obj - timedelta(days=7)

    # Extract date components
    day_from = date_from_obj.day
    month_from = date_from_obj.month
    year_from = date_from_obj.year

    day_to = date_obj.day
    month_to = date_obj.month
    year_to = date_obj.year

    # Construct the URL
    url = f"{base_url}?DateFrom={month_from}%2F{day_from}%2F{year_from}&datepicker-month-select={month_from-1}&datepicker-year-select={year_from}&DateTo={month_to}%2F{day_to}%2F{year_to}&datepicker-month-select={month_to-1}&datepicker-year-select={year_to}"

    return url

# Function to extract news URLs from the Kitchener Newsroom
def extract_news_urls(base_url: str) -> list:
    all_urls = []
    page_num = 1

    while True:
        # Create the URL for the current page by appending the page number
        url = f"{base_url}&page={page_num}"

        # Fetch the web page
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all anchor tags with the class "newsTitle"
            news_links = soup.find_all("a", class_="newsTitle")

            # Check if there are any news items on the current page
            if not news_links:
                break

            # Extract the URLs and store them in an array
            urls = [link.get("href") for link in news_links]
            all_urls.extend(urls)

            # Move on to the next page
            page_num += 1
        else:
            print(f"Failed to fetch page {page_num}. Status code: {response.status_code}")
            break

    return all_urls

# Function to extract text from a div element
def extract_PRtext(url, div_class):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: Unable to fetch the webpage.")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    divs = soup.find_all('div', class_=div_class)

    text = ""
    for div in divs:
        text += div.get_text(separator=' ')

    return text.strip()

# Function to extract plain text from a webpage
def extract_plain_text(url):
    # Fetch the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: Unable to fetch the webpage.")
        return "None"
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove all the HTML tags
    for tag in soup.find_all():
        if tag.name == "a":
            # Exclude links
            tag.replace_with(tag.text)
        else:
            tag.replace_with(" " + tag.text + " ")
    
    # Remove extra spaces and newlines
    plain_text = " ".join(soup.get_text().split())
    
    return plain_text

#Function to extact date from a div element
def extract_date(url):
    session = HTMLSession()
    response = session.get(url)

    if response.status_code != 200:
        print(f"Error {response.status_code}: Unable to fetch the webpage.")
        return None

    response.html.render()
    soup = BeautifulSoup(response.html.html, 'html.parser')
    date_tag = soup.find('div', class_='blogPostDate')

    if date_tag:
        date_str = date_tag.get_text().replace('Posted on ', '').strip()
        date = datetime.strptime(date_str, "%A, %B %d, %Y")
        unix_timestamp = int(date.timestamp())
        return unix_timestamp
    else:
        print("Date tag not found.")
        return None

# Function to generate search phrases using OpenAI's GPT-3.5 API
def generate_search_phrases(paragraph, max_attempts=3, num_phrases=3):
    attempt = 0
    while attempt < max_attempts:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant, meant to be used by a journalist to help them find relevant articles for a story. You are given a press release and you need to generate a list of search phrases to use to find relevant articles. You only use the press release to help you generate the search phrases."},
                {"role": "user", "content": f"Generate a JSON object containing a list of {num_phrases} search phrases given the press release: {paragraph}. The JSON object should have a key 'search_phrases' with the list of phrases as its value."}
            ],
            max_tokens=150,  # Limit the response to avoid incomplete JSON strings
            temperature=0.2 # Making the response more deterministic
            )

            # Extract and return search phrases
            assistant_response = response.choices[0].message['content']

            phrases_json = json.loads(assistant_response)
            return phrases_json['search_phrases']
        except Exception as e:
            print(f"Attempt {attempt + 1}: error {e}. Retrying...")
            sleep(30) # Wait for 30 seconds before trying again
            attempt += 1

def is_valid_source(article_url, allowed_sources):
    for source in allowed_sources:
        if article_url.startswith(source):
            return True
    return False

# Function to filter relevant articles with a relevance score of 75% or higher
def filter_relevant_articles(articles, paragraph):
    # Allowed source URLs
    allowed_sources = [
        "https://kitchener.ctvnews.ca/",
        "https://www.therecord.com/news/waterloo-region/",
        "https://www.cbc.ca/news/canada/kitchener-waterloo/",
        "https://www.kitchenerpost.ca/",
        "https://globalnews.ca/kitchener/",
        "https://kitchener.citynews.ca/"
    ]

    relevant_articles = []
    for article in articles:
        try:
            content = extract_plain_text(article["url"])
        except:
            content = "none"
            continue
        relevance_score = calculate_relevance_score(paragraph, content)
        if is_valid_source(article["url"], allowed_sources):
            article["relevance_score"] = relevance_score
            relevant_articles.append(article)
    return relevant_articles

# Function to write the output in markdown format
def write_output_to_file(filename, articles, search_phrase, paragraph, PRSentimentScore):
    # Add new content to the file
    with open(filename, "a") as file:
        file.write(f"## Search Phrase: {search_phrase}\n\n")
        for i, article in enumerate(articles):
            title = article["name"]
            desc = article["description"]
            url = article["url"]
            relevance_score = article["relevance_score"]
            file.write(f"### {i + 1}. {title}\n\n")
            file.write(f"**Sentiment Score:** {calculate_sentiment_score(extract_plain_text(url)):.2f}%\n")
            file.write(f"**Sentiment Difference:** {(calculate_sentiment_score(extract_plain_text(url)) - PRSentimentScore):+.2f}% | **Relevance Score:** {relevance_score:.2f}%\n\n")
            file.write(f"**Description:** {desc}\n\n")
            file.write(f"**URL:** {url}\n\n")
        file.write("---\n\n")  # Add a separator between search results

# Function to generate a title for the press release
def gen_title(paragraph):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Generate a title for the following press release: {paragraph} (5 words or less)"}
        ],
            max_tokens=50,  # Limit the response to avoid incomplete JSON strings
            temperature=0.2 # Making the response more deterministic
        )

        # Extract and return search phrases
    summary = response.choices[0].message['content']
    return summary

# Find all the related articles for a press release 
def MediaSearch(url):
    #Setup Bing Search API & OpenAI API
    bing_subscription_key = "649101c60ac247e39c46f5aaef08039d"
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key" : bing_subscription_key}
    
    # Get text from Kitchener's website
    div_class = "iCreateDynaToken"
    paragraph = extract_PRtext(url, div_class)

    # Generate a title and sentiment score for the press release
    title = gen_title(paragraph)
    PR_sentiment = calculate_sentiment_score(paragraph)
    
    # Create the output file if it doesn't exist
    filename = f"Reports/{title}.md"
    with open(filename, "w") as file:
        file.write(f"# {title}\n")
        file.write(f"## PR Sentiment Score: {PR_sentiment:.2f}%\n\n")

    # Call the function and print the search phrases
    search_phrases = generate_search_phrases(paragraph)
    totalArticles = 0

    # Initialize a set to store unique article URLs
    unique_articles = set()

    for phrase in search_phrases:
        params  = {
                    "q": (phrase + " " + "Kitchener"),
                    "textDecorations": True,
                    "textFormat": "HTML",
                    "mkt": "en-CA",
                    "freshness": "Month"
                    }

        response = requests.get(search_url, headers=headers, params=params)
        # Wait before raising the status or making another request
        time.sleep(3)  # Adjust the delay (in seconds) based on the API's rate limits
        
        response.raise_for_status()
        search_results = response.json()

        articles = search_results["value"]  # Get all articles
        relevant_articles = filter_relevant_articles(articles, paragraph)  # Filter relevant articles
        totalArticles += len(relevant_articles)
        
        # Add the URLs of the relevant articles to the unique_articles set
        for article in relevant_articles:
            unique_articles.add(article["url"])

        write_output_to_file(filename, relevant_articles, phrase, paragraph, PR_sentiment)
    
    with open(filename, "a") as file:
        file.write(f"# TOTAL ARTICLES:{len(unique_articles)}\n\n")

#----------MAIN PROGRAM------------
def main():
    date = datetime.today().strftime('%m/%d/%Y')

    lastweekNewsroom_url = construct_url(date)
    newsURLs= extract_news_urls(lastweekNewsroom_url)
    
    for eachURL in newsURLs:
        MediaSearch(eachURL)

    # Prepare the email
    recipients = ['diptob@kitchener.ca', 'dbiswas@uoguelph.ca']
    subject = 'Media Search Reports for ' + date
    body = 'Attached are the generated media search reports.'

    # Find report files in the Reports/ folder
    report_dir = 'Reports/'
    report_files = [os.path.join(report_dir, file) for file in os.listdir(report_dir) if os.path.isfile(os.path.join(report_dir, file))]

    # Send the email with attached files
    send_email(report_files, recipients, subject, body)

    # Delete the report files
    delete_files(report_files)

if __name__ == "__main__":
    main()