import openai
import json 
import requests
import time
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate the relevance score
def calculate_relevance_score(paragraph, article_description):
    vectorizer = TfidfVectorizer()
    texts = [paragraph, article_description]
    tfidf_matrix = vectorizer.fit_transform(texts)
    relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return relevance_score[0][0]

# Function to extract text from a div element
def extract_text_from_div(url, div_class):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    divs = soup.find_all('div', class_=div_class)

    text = ""
    for div in divs:
        text += div.get_text(separator=' ')

    return text.strip()

# Function to generate search phrases
def generate_search_phrases(paragraph):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, meant to be used by a journalist to help them find relevant articles for a story. You are given a press release and you need to generate a list of search phrases to use to find relevant articles. You only use the press release to help you generate the search phrases."},
            {"role": "user", "content": f"Generate a JSON object containing a list of 3 search phrases given the press release: {paragraph}. The JSON object should have a key 'search_phrases' with the list of phrases as its value."}
        ],
        max_tokens=150  # Limit the response to avoid incomplete JSON strings
    )

    # Extract and return search phrases
    assistant_response = response.choices[0].message['content']
    try:
        phrases_json = json.loads(assistant_response)
        return phrases_json['search_phrases']
    except json.JSONDecodeError:
        print("The AI model did not return a valid JSON response.")
        return []

# Function to write the output in markdown format
def write_output_to_file(articles, search_phrase, paragraph):
    filename = "search_results.md"
    with open(filename, "a") as file:
        file.write(f"## Search Phrase: {search_phrase}\n\n")
        for i, article in enumerate(articles):
            title = article["name"]
            desc = article["description"]
            url = article["url"]
            relevance_score = calculate_relevance_score(paragraph, desc)
            file.write(f"### {i + 1}. {title}\n\n")
            file.write(f"**Relevance Score:** {relevance_score:.2f}\n\n")
            file.write(f"**Description:** {desc}\n\n")
            file.write(f"**URL:** {url}\n\n")
        file.write("---\n\n")  # Add a separator between search results

#----------MAIN PROGRAM------------
#Setup Bing news Search API
subscription_key = "649101c60ac247e39c46f5aaef08039d"
search_url = "https://api.bing.microsoft.com/v7.0/news/search"
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

# Set your API key
openai.api_key = "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"

# Get text from Kitchener's website
url = "https://www.kitchener.ca/en/news/heritage-bridge-in-victoria-park-temporarily-closing-for-repairs.aspx"
div_class = "iCreateDynaToken"
paragraph = extract_text_from_div(url, div_class)
print(paragraph, "\n")


# Call the function and print the search phrases
search_phrases = generate_search_phrases(paragraph)
print("Search phrases:")
for phrase in search_phrases:
    print(f"- {phrase}")
    params  = {"q": phrase,
               "textDecorations": True,
               "textFormat": "HTML",
               "cc": "CA"}

    response = requests.get(search_url, headers=headers, params=params)
    # Wait before raising the status or making another request
    time.sleep(3)  # Adjust the delay (in seconds) based on the API's rate limits
    
    response.raise_for_status()
    search_results = response.json()

    articles = search_results["value"][:5]  # Get the top 5 articles
    write_output_to_file(articles, phrase, paragraph)