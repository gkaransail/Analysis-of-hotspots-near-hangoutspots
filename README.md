# ANALYSIS OF HOTSPOTS NEAR HANGOUTSPOTS IN A PARTICULAR LOCATION

Authors:  **Aiyngaran Chokalingam**, **Raghav Ravichandran**.

YouTube Video:  [Link](https://youtu.be/dgF2n3G76yk)

---

## Introduction

##### The purpose of this project is to visually display a list of venues within a certain radius along with its review, provided a particular event location by the user from any part of the world

The live data retrieved from Eventbrite API
- *List of Events in a given location*
- *Latitudes and Longitudes of the particular event location*

The live data retrieved from Foursquare API
- *List of Hangout spots around the particular event location*

The live data retrieved from Google API
- *Details such as reviews, address, etc. of the hangout spots based on the latitude and longitude obtained from Foursquare API*

The data are obtained from the respective API's provide us the necessary base to conduct our analysis. Using this data we would be able to help people attending the event to look for various hangout spots around that event. This would help people from different places in the world to plan their itinerary for the trip and make decisions on which hangout spots to visit based on the review scores

- *Based on the data obtained from the above-mentioned API's we provide a list of events and hangout spots along with it's reviews around the selected event. We analyze the polarity of the reviews and provide review scores through sentiment analysis. Visually display the hangout spots on a map along with their distance from the event location. Furthermore, we provide an overview of the reviews which helps the user to make a decision*

---

## References
- Source code was adapted from [Introduction to the Eventbrite platform](https://www.eventbrite.com/platform/docs/introduction)
*, 
[A breif guide to using Foursquare API](https://medium.com/@aboutiana/a-brief-guide-to-using-foursquare-api-with-a-hands-on-example-on-python-6fc4d5451203)
,*
[Google Places API Documentation](https://developers.google.com/places/web-service/details)
- The code retrieves data from [Eventbrite](https://www.eventbrite.com/)
*,
[Foursquare](https://foursquare.com/)
,*
[Google Places API](https://developers.google.com/apis-explorer/#p/)
- The code references [Medium](https://medium.com/@aboutiana/a-brief-guide-to-using-foursquare-api-with-a-hands-on-example-on-python-6fc4d5451203)
 ,
 [wordcloud](https://www.datacamp.com/community/tutorials/wordcloud-python)
 *,
 [dropdownlist](https://www.geeksforgeeks.org/python-gui-tkinter/)
 ,*
 [sentiment analysis](https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk)
---

## Requirements
There are some general library requirements for the project and some of which are specific to individual methods. The general requirements are as follows.
- *numpy*
- *pandas*
- *geopy*
- *sklearn*
- *seaborn*
- *spacy*
- *gensim*
- *WordCloud*
- *API keys for events from Eventbrite, hangout spots from Foursquare and place location API from Google;*

**NOTE**:  It is recommended to use Anaconda distribution of Python.

---

## Explanation of the Code

The code, `event_hangoutspots.ipynb`, begins by importing necessary Python packages:
```
import math
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 200)
from pandas.io.json import json_normalize
import geopy.geocoders
from geopy.geocoders import Nominatim
import geopy.distance
import requests
import en_core_web_sm
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment.util import *
import gensim
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import folium
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pyLDAvis
import pyLDAvis.gensim
from tkinter import *
```

- *NOTE:  use pip install "name of the library" in anaconda prompt to install the library files which are not predefined in python. Example pip install nltk*

## Fetching Input

Initially, we get the city name as input from the user.

```
city = str(input("Enter the city Name: "))
```
![City](https://user-images.githubusercontent.com/47163552/57921082-85e8be00-786a-11e9-9e9c-cb5621c14012.JPG)

Now, we get the latitude and longitude of the city mentioned by the following code.
```
geopy.geocoders.options.default_user_agent = 'aiyngara'
nom = Nominatim()
location=nom.geocode(city, timeout=5)
lat_1=location.latitude
long_1= location.longitude
```
We then import data from [Eventsbrite]. The latitude and longitude of the city is given as input to Eventbrite API to fetch the list of different events happening in that location.
```
url_1 = "https://www.eventbriteapi.com/v3/events/search?location.longitude={}&location.latitude={}&expand=venue?&token=XXXXXXXXXXXXXXXX".format(long_1, lat_1)
results_1 = requests.get(url_1).json()
event = results_1["events"]
events_table = json_normalize(event)
events = []
venue_id = []
for i in event:
    events.append(i["name"]["text"])
    venue_id.append(i["venue_id"])
x = {"Events" : events, "Venue_id" : venue_id}
x = pd.DataFrame(x)
```
### Creating a drop down list

using tkinter a dropdown list of different events is created. This would help the user to chose the event of their interest. With the help of the below code, we also try to fetch the venue Id for the particular event location.

```
y = x.Events.tolist()

master = Tk()

variable = StringVar(master)
variable.set("Top Events") # default value

w = OptionMenu(master, variable, *(y)) 
w.pack()
venue_id = []
def ok():
    venue_id.append(x.Venue_id[x["Events"] == str(variable.get())].values[0])

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()
```
![Dropdown list 1](https://user-images.githubusercontent.com/47163552/57921052-736e8480-786a-11e9-89c4-85e87c98caef.JPG)

**NOTE**: After selection of the required event, click ok and close the drop-down list box.


With the help of the below code, we get the latitude and longitude of the particular event location based on the venue Id of the event location selected by the user.

```
url_2 = "https://www.eventbriteapi.com/v3/venues/{}/?token=***".format(venue_id[0])
results_2 = requests.get(url_2).json()
lat_2 = results_2['address']['latitude']
long_2 = results_2['address']['longitude']

```
Now, we use the latitude and longitude obtained to fetch the list of 20 hotspots from FourSquare API with in 1000 meter radius from the particular event location using the code below.

```
url_3 = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&r={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    '20180604', 
    lat_2, 
    long_2, 
    1000, 
    20)
results_3 = requests.get(url_3).json()

```

Based on the response obtained from foursquare about the 20 hotspots. We filter out name and coordiantes for each hotspots and store them in seperate list using the follwoing code.
```
venues = results_3['response']['groups'][0]['items']
nearby_venues = json_normalize(venues) 
filtered_columns = ['venue.name', 'venue.id', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = nearby_venues.loc[:, filtered_columns]
list = []
i = 0
while i < len(nearby_venues['venue.categories']):
    x = nearby_venues['venue.categories'].apply(pd.Series)
    list.append(x[0][i]['name'])
    i = i+1

nearby_venues['venue.categories'] = list
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
near_ven = nearby_venues["name"].values.tolist()
list_1 = nearby_venues["name"].tolist()
list_2 = nearby_venues["lat"].tolist()
list_3 = nearby_venues["lng"].tolist()

```
Here, using the name and coordinates obtained from Foursquare API, we get the venue Id of all the 20 hotspots from google API.

```
k=0
venue_id = []

while(k < len(list_1)):
    url_4 = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={}&inputtype=textquery&fields=photos,formatted_address,name,opening_hours,place_id&locationbias=circle:2000@{},{}&key=".format(list_1[k],list_2[k],list_3[k])
    results_4 = requests.get(url_4).json()
    venue_id.append(results_4['candidates'][0]['place_id'])
    k=k+1
```
Using the venue Id obtained hotspots, we now fetch reviews, ratings and other details of the 20 hotspots from google API. 
```
kk=0
venue_rev = {}
latitude = []
longitude = []
while(kk < len(venue_id)):
    url_5 = "https://maps.googleapis.com/maps/api/place/details/json?placeid={}&fields=name,rating,review,formatted_phone_number&key= ".format(venue_id[kk])
    results_5 = requests.get(url_5).json()
    if(len(results_5["result"]) > 1):
        latitude.append(list_2[kk])
        longitude.append(list_3[kk])
        reviews = results_5["result"]["reviews"]
        reviews_table = json_normalize(reviews)
        venue_rev[list_1[kk]]=reviews_table["text"].tolist()
    kk=kk+1
```
Getting the venue names as a list
```
venues = [*venue_rev]
```
Find the distance of Hangout Spots from the event location in meteres.
```
coordinates_1 = (float(lat_2), float(long_2)) #The event location
distance = []
for coord in range(len(latitude)):
    coordinates_2 = (latitude[coord], longitude[coord])
    distance.append(math.ceil(geopy.distance.vincenty(coordinates_1, coordinates_2).km*1000))
 
``` 
## Data Analysis

### Sentiment Analysis

Converting every review into a numerical value with a maximum of 5 points based on the positive and negative scores obtained from SentimentIntensityAnalyzer and stored as ratings_weighted
```
sentance = pd.DataFrame(venue_rev)
score = []
length_review = []
for names in venues:
    score.append(sentance[names].tolist())
for length in range(len(score)):
    length_review.append(len(score[length]))
pos_score = []
neg_score = []
neu_score = []
review_score_pos=[]
review_score_neg=[]
review_score_neu=[]
sid = SentimentIntensityAnalyzer()
gk=0

while(gk<len(score)):
    positive_Score_list=[]
    negative_Score_list=[]
    neutral_Score_list=[]
    gh=0   
    
    while(gh<len(score[gk])):
        ss = sid.polarity_scores(score[gk][gh])
        for k,j in ss.items():
            if(k=='pos'):
                positive_Score_list.append(j)
            if(k=='neg'):
                negative_Score_list.append(j)
            if(k=='neu'):
                neutral_Score_list.append(j)
            
        gh=gh+1
    pos_score.append(positive_Score_list)
    neg_score.append(negative_Score_list)
    neu_score.append(neutral_Score_list)
    review_score_pos.append((sum(pos_score[gk])/length_review[gk])*100)
    review_score_neg.append((sum(neg_score[gk])/length_review[gk])*100)
    review_score_neu.append((sum(neu_score[gk])/length_review[gk])*100)
    gk=gk+1
```
Calculating the rating scores based on the review. We ignore the neutral part and obtain the positive percentage by formula positive score/(positive score+negative score).
```
rev_score = np.array(review_score_pos)/(np.array(review_score_neg)+ np.array(review_score_pos)) * 5
weighted_ratings = (0.5*np.array(ratings)) + (0.5* rev_score)
ratings_weighted = np.around((weighted_ratings),2).tolist()
```
## Creating Map using folium

Plotting the hangoutspots around a selected event
```
venues_map = folium.Map(location=[float(lat_2), float(long_2)], zoom_start=15, tiles="CartoDB dark_matter")
folium.Marker(location=[float(lat_2), float(long_2)], icon=folium.Icon(color='blue', icon='ok-sign')).add_to(venues_map)
for lat, lng, label, dist, rat in zip(latitude, longitude, venues, distance, ratings_weighted):
    folium.CircleMarker(
        [lat, lng], radius = rat**2 , color = "green", popup ="Name : " + label + ", distance : " + str(dist) + " meters" + "ratings : " + str(rat) , fill_color = '#ef6c34', fill_opacity=0.3
    ).add_to(venues_map)
venues_map
```
![MAP](https://user-images.githubusercontent.com/47163552/57920951-3f935f00-786a-11e9-8340-5554bdf5ce12.JPG)
```
```
creating a dictionary of venue names, ratings(from reviews), actual ratings, distance and reviews
```
dictionary = {"venue name" : venues, "ratings" : ratings_weighted, "actual ratings" : ratings, "Distance from the event in meters" : distance, "reviews" : score}
final = pd.DataFrame(dictionary)
```
Show the shortest hangout spot from the event location.
```
shortest = final[final["Distance from the event in meters"] == final["Distance from the event in meters"].min()]
shortest
```
Show the best hangout spot from the event location based on the ratings
```
best = final[final["ratings"] == final["ratings"].max()]
best
```
Creating a dropdown list of popular hangoutspots/venues around the event location to make a selection
```
reviews_1 = [*venue_rev]

master = Tk()

variable = StringVar(master)
variable.set("Popular Venues") # default value

w = OptionMenu(master, variable, *(reviews_1)) 
w.pack()
selected_name = []
def ok():
    selected_name.append(str(variable.get()))

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()
venue_name = selected_name[0]
```
**NOTE**: After selection of the required venue, click ok and and close the drop down list box.

Calculating the review scores
```
positive_Score_list_1=[]
negative_Score_list_1=[]
neutral_Score_list_1=[]

length_review = len(sentance[venue_name])
review_score=[]

for dum in sentance[venue_name]:
    ss_1 = sid.polarity_scores(dum)
    #print(ss_1)
    for l,m in ss_1.items():
        if(l=='pos'):
            positive_Score_list_1.append(m)
        if(l=='neg'):
            negative_Score_list_1.append(m)
        if(l=='neu'):
            neutral_Score_list_1.append(m)

review_score.append((sum(positive_Score_list_1)/length_review)*100)
review_score.append((sum(negative_Score_list_1)/length_review)*100)
review_score.append((sum(neutral_Score_list_1)/length_review)*100)
```
### Creating pie chart to plot the review scores.

Pie chart displaying the percentage of positive, negative and neutral scores of all the reviews of a selected hangoutspot
```
color = ["green", "red", "blue"]
labels = ["Positive", "Negative", "Neutral"]
trace = go.Pie(labels=labels, values=review_score)
plotly.offline.init_notebook_mode(connected = True)
plotly.offline.iplot([trace], filename='basic_pie_chart')
```

![Pie chart](https://user-images.githubusercontent.com/47163552/57920997-55088900-786a-11e9-8390-b769d7249cae.JPG)

Creating a function to plot the frequency of words.
```
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    fdist_keys=[]
    fdist_values=[]
    for key in fdist.keys():
        fdist_keys.append(key)
    for values in fdist.values():
        fdist_values.append(values)  
    
    words_df = pd.DataFrame({'word':fdist_keys, 'count':fdist_values})
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
```
Remove unwanted characters, numbers and symbols.
```
sentance[venue_name] = sentance[venue_name].str.replace("[^a-zA-Z#]", " ")
```
Downloading the 'stopwords' (run once)
```
stop_words = stopwords.words('english')
```
function to remove 'stopwords'
```
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

sentance[venue_name] = sentance[venue_name].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

reviews = [remove_stopwords(r.split()) for r in sentance[venue_name]]

# make entire text lowercase
reviews = [r.lower() for r in reviews]
```
import en_core_web_sm
```
nlp = en_core_web_sm.load()
```
A function for lemmatizing (filtering nouns and adjectives)
```
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
  
```
Plotting mostfrequent words after lemmatizing
```
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

sentance[venue_name] = reviews_3
freq_words(sentance[venue_name], 20)
```
![bargraph](https://user-images.githubusercontent.com/47163552/57920858-14107480-786a-11e9-861b-274be076bb02.PNG)

```
Create and generate a word cloud image:
```
```
joined_reviews = " ".join([t for t in reviews])
text = joined_reviews

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

![wordcount](https://user-images.githubusercontent.com/47163552/57920784-e1667c00-7869-11e9-98d2-0fad7c0fbaaf.PNG)

---

## How to Run the Code
*Provide step-by-step instructions for running the code.  For example, I like to run code from the terminal:*
1. Register in Eventsbrite, Foursquare, and Google for username and API key as per the instructions provided in the website

2. Create a private API key and password for fetching the necessary data.

3. Ensure that you have installed necessary Python packages.

4. Download the ipynb file and store it in the jupyter working directory.

5. The program can be executed by running each and every cell of the jupyter notebook or can be run completely by using the run all option.

---

## Results from your Analysis
- *Decision:*

  1. Based on the data fetched from the API. We find the top 20 hangout spots around the event's location.
  2. We find the review score for each hangout spot and display it in a map along with the distance of hangout spots from the event's location.
  

- *Result:*
     *Based on the decisions mentioned above, People who are looking to go for an event in any place of the world can plan for it in an efficient way by deciding the places to visit during the trip beforehand.*
     

## Future Scope

- *Here the review data provided by the Places API were restricted to a few numbers. If provided with more review data, the sentiment analysis to find review scores would be more accurate.*

- *If provided with more time and data, a behavioural analysis could be done on the reviews to tell which kind of people visit or are interested in a particular hangout spot. This could help the user to determine if the place is of their interest.*

- *As of now we have displayed the most occurring adjectives and nouns. If given time, we will be able to provide more accurate results on the positive and negative aspects of a hangout spot.*
