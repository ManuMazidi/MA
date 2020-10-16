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
    print(common_words)

# Unique words that only appear once per PDF
# unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
# print (unique_words)

# Named Entitiy Recognition (as comment as I did not use it but might be useful for certain tasks)
#for text in clean_docs:
#    for ent in text.ents:
#        print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))