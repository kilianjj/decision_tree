"""
File for extracting and saving volume of text data from wikipedia pages
Author: Kilian Jakstis
"""

import requests
from bs4 import BeautifulSoup
import re

english_urls = ["https://en.wikipedia.org/wiki/Computer", "https://en.wikipedia.org/wiki/Microsoft",
                "https://en.wikipedia.org/wiki/Narcissism", "https://en.wikipedia.org/wiki/Attention",
                "https://en.wikipedia.org/wiki/Cannabis_(drug)", "https://en.wikipedia.org/wiki/Computer_font",
                "https://en.wikipedia.org/wiki/Project_management""https://en.wikipedia.org/wiki/English_language"]
english_test_urls = ["https://en.wikipedia.org/wiki/English_people",
                     "https://en.wikipedia.org/wiki/Irreligion"]
dutch_urls = ["https://nl.wikipedia.org/wiki/Biomedische_technologie",
              "https://nl.wikipedia.org/wiki/Itali%C3%AB", "https://nl.wikipedia.org/wiki/One_World_Trade_Center",
              "https://nl.wikipedia.org/wiki/Middellandse_Zee",
              "https://nl.wikipedia.org/wiki/Ontstaansgeschiedenis_van_het_Wilhelminakanaal",
              "https://nl.wikipedia.org/wiki/Kredietcrisis", "https://nl.wikipedia.org/wiki/Beleg_van_Deventer_(1578)",
              "https://nl.wikipedia.org/wiki/Slag_om_de_Ardennen"]
dutch_test_urls = ["https://nl.wikipedia.org/wiki/Geschiedenis_van_de_Aarde",
                   "https://nl.wikipedia.org/wiki/Susanne_Heynemann"]

def get_data(lang):
    """
    Write wiki text data to an intermediate file
    """
    # training data
    urls = english_urls if lang == 1 else dutch_urls
    file_name = r"C:\Users\jakst\PycharmProjects\lab3\english_data.txt" if lang == 1 \
        else r"C:\Users\jakst\PycharmProjects\lab3\dutch_data.txt"
    all_paragraphs = ""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                s = re.sub(r'[^a-zA-Z\s]', ' ', p.get_text().lower())
                all_paragraphs += s
        else:
            print(f"Failed to fetch {url}")
    all_paragraphs.lower()
    with open(file_name, "w", encoding='utf-8') as file:
        file.writelines(all_paragraphs)
    # test data
    test_urls = english_test_urls if lang == 1 else dutch_test_urls
    test_file = r"C:\Users\jakst\PycharmProjects\lab3\english_test_data.txt" if lang == 1\
        else r"C:\Users\jakst\PycharmProjects\lab3\dutch_test_data.txt"
    all_test = ""
    for url in test_urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                s = re.sub(r'[^a-zA-Z\s]', ' ', p.get_text().lower())
                all_test += s
        else:
            print(f"Failed to fetch {url}")
    all_test.lower()
    with open(test_file, "w", encoding='utf-8') as file:
        file.writelines(all_paragraphs)

def split_string_into_substrings(text, words_per_substring=15):
    """
    Split large list of words into 15-word sub-lists for training examples
    """
    words = text.split()
    substrings = []
    for i in range(0, len(words), words_per_substring):
        substring = ' '.join(words[i:i + words_per_substring])
        substrings.append(substring)
    return substrings

def populate_examples_english():
    """
    Write formatted english examples to training data file
    """
    with open(r"C:\Users\jakst\PycharmProjects\lab3\examples.txt", "a") as outfile:
        # english examples
        with open(r"C:\Users\jakst\PycharmProjects\lab3\english_data.txt", "r") as eng_file:
            all_english = eng_file.read()
        subs = split_string_into_substrings(all_english)
        for x in subs:
            outfile.write("en|" + x + "\n")
    with open(r"C:\Users\jakst\PycharmProjects\lab3\test_data.txt", 'a') as file:
        with open(r"C:\Users\jakst\PycharmProjects\lab3\english_test_data.txt", 'r') as test:
            all_test = test.read()
        test_obs = split_string_into_substrings(all_test)
        for x in test_obs:
            file.write("en|" + x + "\n")

def populate_examples_dutch():
    """
    Write formatted dutch examples to training data file
    """
    with open(r"C:\Users\jakst\PycharmProjects\lab3\examples.txt", "a") as outfile:
        # dutch examples
        with open(r"C:\Users\jakst\PycharmProjects\lab3\dutch_data.txt", "r") as datafile:
            all_dutch = datafile.read().strip()
        subs = split_string_into_substrings(all_dutch)
        for z in subs:
            outfile.write("nl|" + z + "\n")
    with open(r"C:\Users\jakst\PycharmProjects\lab3\test_data.txt", 'a') as file:
        with open(r"C:\Users\jakst\PycharmProjects\lab3\dutch_test_data.txt", 'r') as test:
            all_test = test.read()
        test_obs = split_string_into_substrings(all_test)
        for x in test_obs:
            file.write("nl|" + x + "\n")

if __name__ == "__main__":
    get_data(1)
    get_data(0)
    populate_examples_english()
    populate_examples_dutch()
