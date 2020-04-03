#IMDB
#extracting the reviews 
from selenium import webdriver
browser = webdriver.Chrome("C:\\Users\\dell\\chromedriver")
help (webdriver)
from bs4 import BeautifulSoup as bs
page = "https://www.imdb.com/title/tt7430722/reviews?ref_=tt_ql_3"
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
browser.get(page)

import time
reviews = []
i=1
while (i>0):
    #i=i+25
    try:
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"')
        button.click()
        time.sleep(5)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
    
ps = browser.page_source
soup=bs(ps,"html.parser")
reviews = soup.findAll("div",attrs={"class","text"})

for i in range(len(reviews)):
    reviews[i] = reviews[i].text

# Creating a data frame 
import pandas as pd
movie_reviews = pd.DataFrame(columns = ["reviews"])
movie_reviews["reviews"] = reviews    

movie_reviews.to_csv("movie_reviews.csv",encoding="utf-8")

import re
from nltk.corpus import stopwords
reviews_war = ' '.join(reviews)


# Removing unwanted symbols incase if exists
movie_rev_string = re.sub("[^A-Za-z" "]+"," ",reviews_war).lower()
movie_rev_string = re.sub("[0-9" "]+"," ",reviews_war)
#movie_rev_string = re.sub("[.]+","[,]+","-",reviews_war)


# words that contained in war_ reviews
movie_reviews_words = movie_rev_string.split(" ")    

with open("C:\\Users\\dell\\Desktop\\text mining\\stop_words.txt","r") as sw:
    stopwords = sw.read()
#stop_words = stopwords.words('english')

stopwords = stopwords.split("\n")


temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

movie_reviews_words = [w for w in movie_reviews_words if not w in stopwords]

# Joinining all the reviews into single paragraph 
war_rev_string = " ".join(movie_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_war = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(war_rev_string)

plt.imshow(wordcloud_war)


# positive words # Choose the path for +ve words stored in system
with open("C:\\Users\\dell\\Desktop\\text mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("C:\\Users\\dell\\Desktop\\text mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]


# negative word cloud
# Choosing the only words which are present in negwords
war_neg_in_neg = " ".join ([w for w in movie_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(war_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
war_pos_in_pos = " ".join ([w for w in movie_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(war_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
