""" This program performs TF-IDF on a series of job postings to output key terms within
each, plus scans each for defined data science key skills.  These key terms are output
and matched to the input resume, producing a full report. """

import string
import os
import collections
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', 500)

##FULL = 1 does not remove stop words, while FULL = 0 does
FULL = 0
##How many total top skills summed over all job postings to output
N = 20
##How many top tf-idf terms to output for each job posting
TOP_COUNT = 12

RESUME = 'Jeff_Cummings_Resume.pdf'

def skill_list():
    """ User defined data science skills set. """
    return set(['data', 'python', 'keras', 'tensorflow', 'sql', 'r', 'hadoop', 'spark',\
     'java', 'sas', 'tableau', 'hive', 'scala', 'aws', 'c++', 'c', 'matlab', 'excel',\
     'nosql', 'linux', 'azure', 'scikit learn', 'sklearn', 'git', 'pandas', 'numpy',\
     'docker', 'mongodb', 'pytorch', 'pig', 'javescript', 'd3', 'caffe', 'machine Learning',\
     'technical', 'Data Science', 'research', 'algorithms', 'training', 'programming',\
     'artificial intelligence', 'computational neuroscience', 'visual basic', 'technical projects',\
     'gcp', 'virtual machines', 'design', 'docker', 'engineering', 'project management',\
     'data sets', 'scripting', 'statistics', 'experiments', 'big data', 'computer science',\
     'architecture', 'scripting language', 'collaborate', 'quantitative', 'impact', 'negotiating',\
     'decision making', 'safety', 'data sources', 'economics', 'analytics', 'geometry', 'physics',\
     'academia', 'math', 'software engineering', 'analytical', 'systems engineering', 'video',\
     'metrics', 'cloud', 'data visualization', 'calculus', 'mathematics', 'gis', 'mining',\
     'lab work', 'azure', 'visualization', 'kubernetes', 'devops', 'ai', 'statistical analysis',\
     'programming', 'apache', 'predictive analytics', 'project delivery', 'git', 'version control',\
     'collaborative', 'technology trends', 'entrepreneurial', 'passion', 'high quality',\
     'hands on', 'best practices', 'passionate', 'motivated', 'communication skills',\
     'self motivated', 'collaboratively', 'impact', 'track record', 'computer vision',\
     'deep learning', 'ml', 'dl', 'etl', 'image processing', 'audio', 'cnn', 'rnn', 'mlp',\
     'neural network', 'reinforcement', 'phd', 'xgboost', 'support vector machines', 'svm',\
     'graph', 'train', 'matplotlib', 'clustering', 'regression', 'classification', 'forecasting',\
     'time series', 'tree', 'augmentation', 'spatial', 'convolutional neural network',\
     'sustainability', 'efficiency', 'energy', 'astronomy', 'nlp', 'smote', 'monte carlo',\
     'simulation', 'folium', 'api', 'geospatial', 'signal processing', 'speech', 'pipeline',\
     'bash', 'unix', 'osx', 'linux', 'logistic regression', 'linear regression', 'jupyter',\
     'latex', 'json', 'html', 'dash', 'stacking', 'stack', 'bs', 'bachelor', 'validate',\
     'validation', 'css', 'xml', 'team', 'predictive', 'a b test'])

def post_files_search():
    """ Searches current directory for all job posting text files, those with "job_post.txt" """
    os.system('ls *job_post.txt > post_files.txt')
    with open("post_files.txt", 'r') as post_files:
        return post_files.read().split()

def word_processing():
    """ Setup printable characters set, plus words and symbols to be removed before analysis. """
    printable = set(string.printable)
    symbols = "!,\"#$%&()*-./:;<=>?@[\]^_`{|}~\n"
    stop_words = set(stopwords.words('english'))
    stop_words.update(['behind', 'get', 'exceed', 'problems', 'role', 'experience',\
     'process', 'always', 'nice', 'available', 'partner', 'work', 'status', 'etc', 'use',\
      'able', 'change', 'consistently', 'similar'])
    return stop_words, symbols, printable

def text_proc(text_in, symbols, printable):
    """ Remove defined symbols set and non-printable characters, and set remaining to lowercase. """
    for i in symbols:
        text_in = str(np.char.replace(text_in, i, ' '))
    text_in = text_in.replace('  ', ' ')
    text_in = "".join(filter(lambda x: x in printable, text_in))
    return np.char.lower(text_in)

def lemma(doc_in, stop_words, wordnet_lemmatizer):
    """ Define each word's part of speech and lemmatize all them accordingly. """
    doc_in = str(doc_in).split()
    word_list = []
    for word, tag in pos_tag(doc_in):
        if word not in stop_words or FULL == 1:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            word_list.append(wordnet_lemmatizer.lemmatize(word, wntag) if wntag else word)
    return " ".join(word_list), word_list

def post_proc(post_files, stop_words, symbols, printable, wordnet_lemmatizer):
    """ Perform all stages of processing for all job postings. """
    postings = []
    company_list = []
    filtered_postings = []
    postings_length = []
    for file_in in post_files:
        company = file_in.split('_')[0].split('.')
        stop_words.update(company)
        company_list.append(" ".join(company))
        with open(file_in, 'r', encoding='utf-8') as file:
            input_text = file.read()
            postings.append(text_proc(input_text, symbols, printable))
    for job in postings:
        out, post_list = lemma(job, stop_words, wordnet_lemmatizer)
        filtered_postings.append(out)
        postings_length.append(len(post_list))
    return filtered_postings, company_list, stop_words, postings_length

def skill_scanner(title, dictionary, text_in, check, dict2=None):
    """ Scan text_in for keywords in input dictionary, logging their count.  In the special
     case for matching to the resume, we produce a dict2 that only includes keywords that
     appear in at least one job posting (text_in). """
    out = []
    print()
    print(title)
    for key in dictionary:
        dictionary[key] = text_in.count(' '+key+' ')
        if dictionary[key] > 0:
            out.append([key.title() if len(key) > 3 else key.upper(), dictionary[key]])
            if check == 1:
                dict2[key] = 0
    out.sort(key=lambda x: x[1], reverse=True)
    for element in out:
        print(*element)
    if check == 1:
        return dictionary, dict2
    return dictionary

def top_total_skills(skill_dict):
    """ Count and sum each skill count over all scanned job postings, outputting the
     top N most import keywords. """
    counter = collections.Counter()
    for skill_in in skill_dict:
        counter.update(skill_in)
    top_n = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)[:N]
    print()
    print(f"Total Top {N} Skills")
    print(list(zip(*top_n))[0])

def tf_idf(filtered_postings):
    """ Perform TF-IDF analysis on all postings, and output these values as a dataframe. """
    vectorizer = TfidfVectorizer(max_df=0.6)
    vectors = vectorizer.fit_transform(filtered_postings)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=feature_names)

def skill_matcher(index, title, resume_dict, resume_list, skill_dict, words_df):
    """ Calculates both a keyword percentage match, and a TF-IDF score based on each
     resume word's TF-IDF value for a given posting. """
    print()
    match = 0
    tf_idf_score = 0
    print(title)
    for key in resume_dict:
        match += min(resume_dict[key], skill_dict[key])
    match /= sum(skill_dict.values())
    print(f'Skill Match: {round(match * 100, 2)}%')
    for term in resume_list:
        try:
            tf_idf_score += words_df.at[index, term]
        except:
            pass
    print(f'TF-IDF Score: {round(tf_idf_score, 2)}')

def top_tf_idf(words_df, company_list):
    """ Outputs the top (TOP_COUNT) words for each posting based on their TF-IDF value. """
    tops = pd.DataFrame(words_df.apply(lambda x: list(words_df.columns[np.array(x).argsort()[::-1]\
    [:TOP_COUNT]]), axis=1).to_list(), columns=['Top'+str(x) for x in range(1, TOP_COUNT+1)])
    tops.index = company_list
    print(tops)

def main():
    """ The main series of NLP commands. """
    wordnet_lemmatizer = WordNetLemmatizer()

    post_files = post_files_search()
    stop_words, symbols, printable = word_processing()
    filtered_postings, company_list, stop_words, postings_length = post_proc(post_files,\
     stop_words, symbols, printable, wordnet_lemmatizer)

    res_text = extract_text(RESUME)
    res_text = text_proc(res_text, symbols, printable)
    resume_final, resume_list = lemma(res_text, stop_words, wordnet_lemmatizer)
    
    ds_skills = skill_list()
    skill_dict = [dict.fromkeys(ds_skills, 0) for _ in range(len(post_files))]

    resume_dict = {}
    title = []

    for index, post in enumerate(filtered_postings):
        title.append(f'{company_list[index].title() if len(company_list[index]) > 3 else company_list[index].upper()} - Number of Cleaned Words in Posting: {postings_length[index]}')
        skill_dict[index], resume_dict = skill_scanner(title[index], skill_dict[index], post,\
         1, resume_dict)

    top_total_skills(skill_dict)

    resume_dict = skill_scanner("My Resume", resume_dict, resume_final, 0)

    words_df = tf_idf(filtered_postings)

    for index, job in enumerate(title):
        skill_matcher(index, job, resume_dict, resume_list, skill_dict[index], words_df)

    top_tf_idf(words_df, company_list)

main()
