from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk import tokenize
import numpy as np
import networkx as nx
from pickle import load
import string
import argparse
import nltk
nltk.download('punkt')


def clean_lines(lines):
    cleaned = list()
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        line = line.split()
        cleaned.append(' '.join(line))
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned

def read_article_from_file(file_name):
    file = open(file_name, "r")
    filedata = file.read()
    article = clean_lines(tokenize.sent_tokenize(filedata))
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    print("Input Text: ",article)
    return sentences

def read_article_from_pkl(story):
    story_data = story['story']
    highlight_data = story['highlights']
    article = clean_lines(story_data.split(". "))
    sentences = []
    
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    print("Article: ",article)
    print("\nhighlight_data: ",highlight_data, '\n\n')
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix


def generate_summary(file, top_n=5, read_from_pkl=False):
    stop_words = stopwords.words('english')
    summarize_text = []
    if not read_from_pkl:
        sentences =  read_article_from_file(file)
    else:
        sentences = read_article_from_pkl(file)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)  
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    print("\nExtractive Summary: \n", " ".join(summarize_text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-file_path")
    parser.add_argument("-top_n", default=2, type=int)
    args = parser.parse_args()
    generate_summary(args.file_path, args.top_n)
