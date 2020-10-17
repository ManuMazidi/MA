### AUTHOR: Manuel Mazidi
### EMAIL: manuel.mazidi@gmail.com
### GITHUB: https://github.com/ManuMazidi/MA

import PyPDF2 as pdf
import re
import glob
from datetime import datetime
import spacy
import en_core_web_sm


###### Creating a Corpus from PDF Files ######

# Path to folder with all PDF Files
file_paths = glob.glob('/Users/manu/Documents/MA/Data/*')

# create empty list that will be raw corpus and save dates in a list
raw_corpus = []
communication_dates = []

# EXTRACT AND CLEAN TEXT
# the following for-loop extracts dates of the PDFs, number of pages and raw text
for filename in file_paths:

    # get the date of the file
    find_date = re.search("[0-9]{8}", filename)
    pdf_date = datetime.strptime(find_date.group(), '%Y%m%d')
    pdf_datetime = pdf_date.date()
    communication_dates.append(pdf_datetime)

    # read PDF
    pdf_file = open(filename, 'rb')
    read_pdf = pdf.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()

    # get number of pages per file if you need
    # print(number_of_pages)

    # get text by extracting every page and then append it
    page_content = ""  # define variable for using in loop.
    for page_number in range(number_of_pages):
        page = read_pdf.getPage(page_number)
        page_content += page.extractText()  # concate reading pages.

    # clean up extracted text, you may want to add or remove some features depending on PDF
    # this works for FED FOMC Minutes
    new_content = page_content.replace('\n', '')
    new_content = new_content.replace(' o ', '')
    new_content = new_content.replace(' ', ' ')
    new_content = new_content.replace('  ', ' ')
    new_content = new_content.replace('Š', ' ')
    new_content = new_content.replace('™', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('Œ', '-')

    # save clean text in list
    raw_corpus.append(new_content)




# PREPROCESSING OF RAW TEXT
# I use spaCy for this, load their english (en) model
# note that the following line works in jupyter notebook but not directly in PyCharm
# work-around to load in PyCharm: Go to Preferences/Python Interpreter, copy path to python interpreter
# in Terminal run: FULL_PATH_TO_PYTHON_INTERPRETER -m spacy download en
nlp = en_core_web_sm.load()


# Create a pre-processing function
# The following tasks are done in the pre-preprocessing function:
# Lowercases the text, lemmatizes each token, removes punctuation symbols, Removes stop words
# removes digits (you may want to keep them depending on task..)
def is_token_allowed(token):
    '''
         Only allow valid tokens which are not stop words,
         punctuation symbols and digits.
    '''
    # not sure if token.is_digit should be included so that no numbers in output...
    if (not token or not token.string.strip() or token.is_stop or token.is_punct or token.is_digit):
        return False
    return True

def preprocess_token(token):
     # Reduce token to its lowercase lemma form
     return token.lemma_.strip().lower()


# the cleaned pre-processed text in saved in a list 'preprocessed_texts'
preprocessed_texts = []
clean_docs = []     # might be useful later for named-entity recognition
for clean_text in raw_corpus:
    # create doc object
    clean_doc = nlp(clean_text)
    clean_docs.append(clean_doc)
    # preprocessing
    complete_filtered_tokens = [preprocess_token(token) for token in clean_doc if is_token_allowed(token)]

    # save tokens of each text in list
    preprocessed_texts.append(complete_filtered_tokens)

# it might be an advantage to discard all words that only appear once in a text
# the following nested for loop discards all words that only appear once per document
# if you wish to keep words with word count = 1, transform next few lines to comments
# Count word frequencies
from collections import defaultdict

frequency = defaultdict(int)
for text in preprocessed_texts:
    for token in text:
        frequency[token] += 1

preprocessed_texts = [[token for token in text if frequency[token] > 1] for text in preprocessed_texts]
# --- comments until here if you wish to keep words with word count = 1


# it might be interesting to look at the top five words that appear the most in a single PDF
from collections import Counter

for clean_tokens in preprocessed_texts:
    # Remove stop words and punctuation symbols
    word_freq = Counter(clean_tokens)

    # 5 commonly occurring words with their frequencies
    common_words = word_freq.most_common(5)
    #print(common_words)

# Unique words that only appear once per PDF
# unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
# print (unique_words)

# Named Entitiy Recognition (as comment as I did not use it but might be useful for certain tasks)
#for text in clean_docs:
#    for ent in text.ents:
#        print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))



# PREPARE CORPUS FROM 'preprocessed_texts'
import os.path
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def prepare_corpus(doc_clean): #corpus for LSA Model
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix


# create a dictionary object, e.g. if you want to know of how many unique tokens your dictionary from your corpus consists of:
from gensim.corpora import Dictionary
dct = corpora.Dictionary(preprocessed_texts)



# CREATE GENSIM LSA MODEL
def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics, so that optimal number of topics can be evaluated
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


# LSA Model
number_of_topics=4
words=10
#document_list,titles=load_data("","articles.txt")
model=create_gensim_lsa_model(preprocessed_texts,number_of_topics,words)

# Create Plot from LSA Model
start,stop,step=2,12,1
plot_graph(preprocessed_texts,start,stop,step)





# CREATE GENSIM LDA MODEL
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim import matutils
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import pandas as pd
import numpy as np
import pickle
import lda


# Convert texts to lda-specific corpous
lda_corpus = [dct.doc2bow(text) for text in preprocessed_texts]

# fit LDA model
speeches_topics = LdaModel(corpus=lda_corpus,
                           id2word=dct,
                           num_topics=8,
                           passes=5)

# print out first 8 topics
for i, topic in enumerate(speeches_topics.print_topics(8)):
    print (i, topic)


# visualization of topics
#vis_data = gensimvis.prepare(speeches_topics, lda_corpus, dct)
#pyLDAvis.display(vis_data)