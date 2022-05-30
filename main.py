import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import matplotlib.pyplot as plt
import string
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from nltk.corpus import stopwords
stop_words = stopwords.words("russian")
stop_words.append('–')
stop_words.append('-')
import numpy as np
from nltk.book import FreqDist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from tkinter import *
from pprint import pformat
import os.path

def download_pages():
    url = 'https://omgtu.ru/l/?SHOWALL_1=1'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    news_items = soup.find_all('p', class_ = 'news-item')
    news_info = {}
    for item in news_items:
        news_info[item.find('a')['href']] = item.find('b').text
    df = pd.DataFrame(news_info.items(), columns=['href','title'])
    for i, row in df.iterrows():
        file_name=row['href'].replace('/l/?eid=', '')
        if not os.path.exists(f'./news_pages/{file_name}.txt'):
            news_url='https://omgtu.ru' + row['href']
            r=requests.get(news_url)
            with open(f'./news_pages/{file_name}.txt','w', encoding='utf-8') as f:
                f.write(r.text)
    df.to_excel('news_omstu.xlsx')

def get_corpus():
    df = pd.read_excel('news_omstu.xlsx', index_col=0)
    dates, authors, words = [], [], []
    for i, row in df.iterrows():
        file_name = row['href'].replace('/l/?eid=', '')
        with open(f'./news_pages/{file_name}.txt','r', encoding='utf-8') as f:
            content = f.read()
            soup = BeautifulSoup(content, 'lxml')
            news = soup.find('div', class_='news-detail')
            date = news.find('span', class_='news-date-time').text
            dates.append(date)
            try:
                author = news.find('div', {'style' : 'width: 100%; text-align: right;'})
                author = author.find('b').text
                authors.append(author)
            except:
                author = 'Ошибка при парсинге'
                authors.append(author)
            news.span.decompose()
            text = re.sub("^\s+|\n|\t|\r|\s+$", ' ', news.get_text())
            text = text.replace(row['title'], '')
            text = text.replace(author, '')
            text = text[:text.find('Опубликовано: ')]
            count = sum([i.strip(string.punctuation).isalpha() for i in text.split()])
            words.append(count)
            with open(f'./corpus/{file_name}.txt', 'w', encoding='utf-8') as corpus_f:
                corpus_f.write(text)    
    df['date'] = dates
    df['author'] = authors
    df['words count'] = words
    df.to_excel('news_omstu.xlsx')

def date_count():
    df = pd.read_excel('news_omstu.xlsx', index_col=0)
    fig = plt.figure("Графики", figsize=(15, 9.5))
    ax1 = fig.add_subplot(131)
    ax1.set_title("Количество статей за год")
    ax1.set_ylabel("Количество статей")
    ax1.set_xlabel("Год")
    ax2 = fig.add_subplot(132)
    ax2.set_title("Среднее количество слов за год")
    ax2.set_ylabel("Количество слов")
    ax2.set_xlabel("Год")
    ax3 = fig.add_subplot(133)
    ax3.set_title("Максимальное количество слов за год")
    ax3.set_ylabel("Количество слов")
    ax3.set_xlabel("Год")
    dates = list(range(2011, 2023))
    count_news = []
    avg_words = []
    max_words = []
    for date in dates:
        year_df = df.loc[df['date'].str.endswith(str(date))]
        count_news.append(len(year_df))
        avg_words.append(year_df['words count'].mean())
        max_words.append(max(year_df['words count']))
    ax1.bar(dates, count_news)
    ax2.bar(dates, avg_words)
    ax3.bar(dates, max_words)
    plt.show()
    
def lemma(text):
    morph_analyze = morph.parse(text.lower())
    if morph_analyze[0].normal_form not in stop_words:
        return morph_analyze[0].normal_form

def tfidf(word, sentence, docs):      
      tf = sentence.count(word)/len(sentence)    
      idf = np.log10(len(docs)/sum([1 for doc in docs if word in doc if word not in stop_words]))
      return round(tf*idf, 4)

def analysis_tfidf(word):
    df = pd.read_excel('news_omstu.xlsx', index_col=0)
    docs = list(map(lemma,df['title'].values))
    tfidfs = []
    for i, row in df.iterrows():
        title = row['title'].lower().split()
        title = list(map(lemma, title))
        remove_item = None
        title = list(filter(lambda x: x != None, title))
        res = tfidf(lemma(word), title, docs)
        tfidfs.append(res)
    df['tfidf'] = tfidfs
    df = df.sort_values(by='tfidf', ascending=False)
    return df

            
def get_all_texts_list():
    df = pd.read_excel('news_omstu.xlsx', index_col=0)
    texts = []
    for i, row in df.iterrows():
        file_name = row['href'].replace('/l/?eid=', '')
        with open(f'./corpus/{file_name}.txt','r', encoding='utf-8') as f:
            text = f.read()
            text = text.lower().split()
            text = list(map(lemma, text))
            text = list(filter(lambda x: x != None, text))
            texts.append(text)
    return texts

def most_phrases(texts):
    text = []
    for t in texts:
        for w in t:
            text.append(w)
    fdist = FreqDist(text)
    fdist.plot(20, title = 'Самые часто встречающиеся слова')

def cos_similarity(texts):
    docs = [' '.join(t) for t in texts]
    vectorizer = CountVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    fig = plt.figure("Матрица схожести текстов", figsize=(15, 9.5))
    ax = fig.add_subplot(111)
    ax.matshow(csim)
    plt.show()

def button1_click():
    download_pages()
    get_corpus()

def button2_click():
    texts = get_all_texts_list()
    most_phrases(texts)
    cos_similarity(texts)

def button3_click():
    t = input_txt.get(1.0, "end-1c")
    df = analysis_tfidf(t)
    out = ''
    for i, row in df.iterrows():
        if row['tfidf'] > 0:
            out += (row['title'] + ' : ' + str(row['tfidf']) + '\n')
    tfidf_window = Tk()
    tfidf_window.geometry("1000x1000")
    tfidf_window.title("TF-IDF")
    text = Text(tfidf_window,height = 1000, width = 1000)
    text.pack()
    text.insert(1.0, out)
    tfidf_window.mainloop()

def button4_click():
    date_count()

root = Tk()
root.title("Статический анализ новостей ОмГТУ")
root.geometry("300x250")

btn1 = Button(root, text="Загрузить/обновить корпус", height = 2, command = button1_click)
btn2 = Button(root, text="Самые частые слова и матрица схожести", height = 2, command = button2_click)
input_txt = Text(root, height = 2, width = 20)
btn3 = Button(root, text="Анализ релевантности tf-idf (введите слово)", height = 2, command = button3_click)
btn4 = Button(root, text="Графики по годам", height = 2, command = button4_click)

btn1.pack()
btn2.pack()
input_txt.pack()
btn3.pack()
btn4.pack()

def main():
    root.mainloop()

if __name__ == "__main__":
    main()
