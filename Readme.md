# Semantic Media Search using Bing Search API & GPT-3.5

This is the README for the mediaSearch.py program. The purpose of this program is to search and extract relevant articles using AI-generated search phrases based on a given paragraph.

## Libraries

The following libraries are required to run mediaSearch.py:

- openai
- json
- requests
- time
- bs4
- sklearn

The libraries can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run

After installing the required libraries, run the program using the following command:

```bash
python mediaSearch.py
```


## Code Explanation

Here's an overview of the main functions and purpose of the code:

- **calculate_relevance_score(paragraph, article_description):** This function calculates the relevance score of an article description compared to the input paragraph using cosine similarity and TF-IDF.
- **extract_text_from_div(url, div_class):** This function extracts text from a specific div element on a web page by providing the URL and the div class.
- **generate_search_phrases(paragraph):** This function uses OpenAI's GPT-3.5-turbo model to generate a list of 3 search phrases based on the input paragraph. The output is a JSON object with a key 'search_phrases' containing the list of phrases.
- **write_output_to_file(articles, search_phrase, paragraph):** This function writes the output in markdown format, including the search phrases, article titles, descriptions, URLs, and relevance scores.
- **MAIN PROGRAM:** The main part of the program sets up the Bing News Search API and OpenAI's API key. It then extracts the text from Kitchener's website, generates search phrases, and writes the output to a markdown file.

## License

This project is licensed under the terms of the MIT License.
