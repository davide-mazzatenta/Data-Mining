import requests
from lxml import html
import csv
import time
from nltk.corpus import stopwords
import string
import math
import re
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function that is used to process the description of a product or the query
# Takes in input a string, load the italian stopwords, set all the characters to lower case,
# replace / andd | that I saw to be frequent in the descriptions, then add spaces before and 
# after non alphanumerical, spaces and apostrophe characters to separate some strange patterns. 
# Then remove all the puntuaction and split all the words into tokens. 
# All that is used to process the description.
def preprocess_description(description):
    # Manually written words to remove from the description
    remove_keywords = {
        "affordable", "business", "office",
        "wlan", "devices", "dvd", "wifi", "installato", "365", "mouse", "tastiera",
        "supporto", "bluetooth", "espansione", "ricondizionato", "professional",
        "smartphone", "rpm", "•", "|", "【", "】"
    }
    
    stop_words = set(stopwords.words('italian'))
    description = description.lower()
    description = description.replace('/', ' ')
    description = description.replace('丨', ' ')
    description = re.sub(r'([^\w\s’])', r' \1 ', description)
    translator = str.maketrans('', '', string.punctuation)
    description = description.translate(translator)
    tokens = description.split()

    processed_tokens = [
        word for word in tokens
        if word not in stop_words and word not in remove_keywords
    ]

    return ' '.join(processed_tokens)

# This function takes in input the keyword of the amazon product, in this case "computer", 
# the number of pages and the delay between each call to amazon pages. Then builds an header
# and do a get to the amazon page. The response is processed by using lxml and looking at the 
# sections of the html that I found by analizing the web pages of Amazon. Then everything is 
# assembled together. 
def get_amazon_products(keyword, pages, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    }
    products = []
    for page in range(1, pages + 1):
        url = f"https://www.amazon.it/s?k={keyword}&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            tree = html.fromstring(response.content)
            for product in tree.xpath('//div[@data-asin and @data-asin != ""]'):
                if not product.xpath('.//div[@data-component-props and contains(., "sp_hidden")]'):
                    description = product.xpath('.//span[contains(@class, "a-size-base-plus a-color-base a-text-normal")]/text()')
                    description = description[0] if description else "N/A"
                    relative_url = product.xpath('.//a[contains(@class, "a-link-normal s-no-hover s-underline-text s-underline-link-text s-link-style a-text-normal")]/@href')
                    product_url = "https://www.amazon.it" + relative_url[0] if relative_url else "N/A"
                    price = product.xpath('.//span[@class="a-price"]/span[@class="a-offscreen"]/text()')
                    price = price[0] if price else "N/A"
                    is_prime = bool(product.xpath('.//span[@class="aok-relative s-icon-text-medium s-prime"]/i[contains(@aria-label, "Amazon Prime")]'))
                    stars = product.xpath('.//span[@class="a-icon-alt"]/text()')
                    stars = stars[0].split(" ")[0].replace(',', '.') if stars else "N/A"
                    if description == "N/A" or price == "N/A" or product_url == "N/A":
                        continue
                    processed_description = preprocess_description(description)
                    if processed_description is not None:
                        products.append({
                            "description": processed_description,
                            "original_description": description, 
                            "price": price,
                            "prime": is_prime,
                            "url": product_url,
                            "stars": stars
                        })
        else:
            print(f"Failed to retrieve page {page}")
        time.sleep(delay)
    return products

# This function takes in input the list of all the products, with the description processed
# and the name of a file and write them into it with extension .tsv
def save_processed_to_tsv(products, filename="preprocessed_products.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Description", "Price", "Prime", "URL", "Stars"])
        for product in products:
            writer.writerow([
                product["description"],
                product["price"],
                "Yes" if product["prime"] else "No",
                product["url"],
                product["stars"]
            ])

# This function is equal to the previous one exept the fact that it is saved into
#  the file the original description of the products without preprocessing them
def save_non_preprocessed_to_tsv(products, filename="products.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Description", "Price", "Prime", "URL", "Stars"])
        for product in products:
            writer.writerow([
                product["original_description"],
                product["price"],
                "Yes" if product["prime"] else "No",
                product["url"],
                product["stars"]
            ])

# This function takes in input the list of the products with the description processed,
# build an inverted index with them and calcolate the IDF. It iterates over the products,
# splits the words and count them, then builds an inverted index with the term, 
# the document ID as a number derived from the different products and the frequency in that description.  
# Calculate the IDF with the formula.
def build_inverted_index(products):
    inverted_index = defaultdict(list)
    document_frequencies = Counter()
    for doc_id, product in enumerate(products):
        terms = product["description"].split()
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            inverted_index[term].append((doc_id, count))
            document_frequencies[term] += 1
    num_documents = len(products)
    idf = {
        term: math.log(num_documents / (1 + freq))
        for term, freq in document_frequencies.items()
    }
    return inverted_index, idf

# This function saves the inverted index in a file .tsv 
def save_inverted_index(inverted_index, filename="inverted_index.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Term", "Document ID", "Term Frequency"])
        for term, doc_list in inverted_index.items():
            for doc_id, term_freq in doc_list:
                writer.writerow([term, doc_id, term_freq])


# THis function takes the product list and the idf and calculates the TF-IDF by iterating and
# counting the frequency of each term.
def compute_tfidf_vectors(products, idf):
    tfidf_vectors = {}
    for doc_id, product in enumerate(products):
        terms = product["description"].split()
        term_counts = Counter(terms)
        vector = {}
        for term, count in term_counts.items():
            if term in idf:
                tf = count / len(terms)
                vector[term] = tf * idf[term]
        tfidf_vectors[doc_id] = vector
    return tfidf_vectors

# This function takes in input the query, the TF-IDF vectors of the products and the IDF. 
# Process the query with the same metodology as the product description, and calculate the TF-IDF
# for the terms in the query. Then calculate the cosine similarity between the query and the 
# TF-IDF vector built on the product description. Sort them based on the decreasing value of
# similarity and take only the top 10 of them. 
def query_products(query, tfidf_vectors, idf):
    query = preprocess_description(query)

    query_terms = query.split()
    query_vector = {}
    term_counts = Counter(query_terms)

    for term, count in term_counts.items():
        if term in idf:
            tf = count / len(query_terms)
            query_vector[term] = tf * idf[term]

    query_array = np.array([query_vector.get(term, 0) for term in idf.keys()])
    
    similarities = []
    
    for doc_id, doc_vector in tfidf_vectors.items():
        doc_array = np.array([doc_vector.get(term, 0) for term in idf.keys()])
        similarity = cosine_similarity([query_array], [doc_array])[0][0]
        similarities.append((doc_id, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:5]


# This function saves the query results in a file .tsv
def save_query_results(results, products, filename="query_results.tsv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Original Description", "Price", "URL", "Similarity Score"])
        for doc_id, score in results:
            product = products[doc_id]
            writer.writerow([
                product["original_description"],
                product["price"],
                product["url"],
                score
            ])

# Main function
if __name__ == "__main__":
    keyword = "computer"
    num_pages = int(input("Quante pagine vuoi estrarre? "))  
    products = get_amazon_products(keyword, pages=num_pages, delay=2)
    
    save_non_preprocessed_to_tsv(products)
    
    save_processed_to_tsv(products)

    inverted_index, idf = build_inverted_index(products)

    save_inverted_index(inverted_index)

    tfidf_vectors = compute_tfidf_vectors(products, idf)

    query = input("Inserisci i termini della query: ")
    results = query_products(query, tfidf_vectors, idf)

    save_query_results(results, products)
    print("I risultati della query sono stati salvati in query_results.tsv")
