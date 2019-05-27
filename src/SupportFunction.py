from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords 

def show_word(sentiment, category, gram, direction, number, df):
    word = []
    word_list=[]
    count_list=[]
    if gram == 1:
        if sentiment == '':
            df[category].apply(lambda x: word.extend(str(x).split()))
        else:    
            df[df['Sentiment'] == sentiment][category].apply(lambda x: word.extend(str(x).split()))
        words_count = nltk.FreqDist(word)
        if direction == 'Highest':
            most_common_word = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
        else:
            most_common_word = sorted(words_count.items(), key=lambda x: x[1], reverse=False)    
        for word, count in most_common_word:
            word_list.append(word)
            count_list.append(count)    
    elif gram == 2:
        if sentiment == '':
            df[category].apply(lambda x: word.extend(nltk.bigrams(str(x).split())))
        else:    
            df[df['Sentiment'] == sentiment][category].apply(lambda x: word.extend(nltk.bigrams(str(x).split())))       
        words_count = nltk.FreqDist(word)
        if direction == 'Highest':
            most_common_word = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
        else:
            most_common_word = sorted(words_count.items(), key=lambda x: x[1], reverse=False) 
        for word, count in most_common_word:
            word_list.append(' '.join([word[0],word[1]]))
            count_list.append(count)
    data = {'x': word_list[:number], 'y': count_list[:number]}
    return data

def show_word_tfidf(gram, sentiment, category, direction, number, df):
    word_list=[]
    weight_list=[]
    tfv = TfidfVectorizer(analyzer='word', ngram_range = (gram,gram))
    if sentiment == '':
        tfv.fit(df[category])
    else:
        tfv.fit(df[df['Sentiment'] == sentiment][category]) 
    idf_dict = dict(zip(tfv.get_feature_names(),tfv.idf_ ))
    
    if direction == 'Highest':
        weighted_word = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
    else:
        weighted_word= sorted(idf_dict.items(), key=lambda x: x[1], reverse=False)
    for word, weight in weighted_word:
        word_list.append(word)
        weight_list.append(weight)
    data = {'x':word_list[:number], 'y':weight_list[:number]}
    return data 

stops = list(stopwords.words('english'))
stops.extend(['book', 'product', 'movie', 'music', 'album', 'cd'])
stops = set(stops)

def meaningful_word_specific(word):
    letters_only = re.sub("[^a-zA-Z]", " ", word)
    words = letters_only.lower().split()   
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))
