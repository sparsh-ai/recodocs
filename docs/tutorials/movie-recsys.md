---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="GLNC2HYiklWl" -->
# Movie Recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="txUrd30jP6jX" executionInfo={"status": "ok", "timestamp": 1615180365604, "user_tz": -330, "elapsed": 13261, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1641bc32-370a-4d62-ab91-1d84b3e59802"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rounakbanik/the-movies-dataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="VxSAqBRmQErv" executionInfo={"status": "ok", "timestamp": 1615180391940, "user_tz": -330, "elapsed": 10309, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="47f339e4-5ebd-47d5-a3ad-134f53f2abf5"
!unzip the-movies-dataset.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 700} id="OpWMotILQLah" executionInfo={"status": "ok", "timestamp": 1615184027598, "user_tz": -330, "elapsed": 3459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1e98fb57-02ad-42c6-9813-42aa0926ecad"
import pandas as pd
import numpy as np

df = pd.read_csv('movies_metadata.csv')
df.head()
```

<!-- #region id="OXoVndvgQwa9" -->
## Simple Recommender

The choice of a metric is arbitrary. One of the simplest metrics that can be used is the movie rating. However, this suffers from a variety of disadvantages. In the first place, the movie rating does not take the popularity of a movie into consideration. Therefore, a movie rated 9 by 100,000 users will be placed below a movie rated 9.5 by 100 users.
This is not desirable as it is highly likely that a movie watched and rated only by 100 people caters to a very specific niche and may not appeal as much to the average person as the former.

It is also a well-known fact that as the number of voters increase, the rating of a movie normalizes and it approaches a value that is reflective of the movie's quality and popularity with the general populace. To put it another way, movies with very few ratings are not very reliable. A movie rated 10/10 by five users doesn't necessarily mean that it's a good movie.

Therefore, what we need is a metric that can, to an extent, take into account the movie rating and the number of votes it has garnered (a proxy for popularity). This would give a greater preference to a blockbuster movie rated 8 by 100,000 users over an art house movie rated 9 by 100 users.

Fortunately, we do not have to brainstorm a mathematical formula for the metric. As the title of this chapter states, we are building an IMDB top 250 clone. Therefore, we shall use IMDB's weighted rating formula as our metric. Mathematically, it can be represented as follows:
<!-- #endregion -->

<!-- #region id="AS0xSnq_Q-0Z" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="ytYw_CELRKa4" -->
For our recommender, we will use the number of votes garnered by the 80th percentile movie as our value for m. In other words, for a movie to be considered in the rankings, it must have garnered more votes than at least 80% of the movies present in our dataset. Additionally, the number of votes garnered by the 80th percentile movie is used in the weighted formula to come up with the value for the scores.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WrP81j1eQebU" executionInfo={"status": "ok", "timestamp": 1615180690425, "user_tz": -330, "elapsed": 808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="68c0a1e5-e5c8-406f-a4a8-cf4ae1a7c321"
#Calculate the number of votes garnered by the 80th percentile movie
m = df['vote_count'].quantile(0.80)
m
```

<!-- #region id="hyD5cpfNRgSj" -->
We can see that only 20% of the movies have gained more than 50 votes. Therefore, our value of m is 50.

Another prerequisite that we want in place is the runtime. We will only consider movies that are greater than 45 minutes and less than 300 minutes in length. Let us define a new DataFrame, q_movies, which will hold all the movies that qualify to appear in the chart:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FMWkY7frRXBG" executionInfo={"status": "ok", "timestamp": 1615180752650, "user_tz": -330, "elapsed": 1096, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3be0fb96-0bda-4c5d-dcda-d06d8fb20170"
#Only consider movies longer than 45 minutes and shorter than 300 minutes
q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]

#Only consider movies that have garnered more than m votes
q_movies = q_movies[q_movies['vote_count'] >= m]

#Inspect the number of movies that made the cut
q_movies.shape
```

<!-- #region id="qN7-uUb1RoXx" -->
We see that from our dataset of 45,000 movies approximately 9,000 movies (or 20%) made the cut. 

Let's calculate C, the mean rating for all the movies in the dataset:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ViSSRxCTRmJJ" executionInfo={"status": "ok", "timestamp": 1615180827830, "user_tz": -330, "elapsed": 1393, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdccda19-3e3f-4d9a-8f26-b119fdc96ec8"
# Calculate C
C = df['vote_average'].mean()
C
```

<!-- #region id="41D8-wYRSDsK" -->
We can see that the average rating of a movie is approximately 5.6/10. It seems that IMDB happens to be particularly strict with their ratings. Now that we have the value of C, we can go about calculating our score for each movie.

First, let us define a function that computes the rating for a movie, given its features and the values of m and C:
<!-- #endregion -->

```python id="gABUnu0kR4a-"
# Function to compute the IMDB weighted rating for each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 824} id="w3N9aNoaSJVm" executionInfo={"status": "ok", "timestamp": 1615180946859, "user_tz": -330, "elapsed": 1437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="988c9b04-25f2-480a-8de4-e98012b5bbb8"
# Compute the score using the weighted_rating function defined above
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies in descending order of their scores
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 25 movies
q_movies[['title', 'vote_count', 'vote_average', 'score', 'runtime']].head(25)
```

<!-- #region id="UCTFtpQRScMh" -->
We can see that the Bollywood film Dilwale Dulhania Le Jayenge figures at the top of the list. We can also see that it has a noticeably smaller number of votes than the other Top 25 movies. This strongly suggests that we should probably explore a higher value of m.
<!-- #endregion -->

<!-- #region id="VyFD02cwSrRD" -->
## Knowledge-based Recommender

This will be a simple function that will perform the following tasks:
- Ask the user for the genres of movies he/she is looking for
- Ask the user for the duration
- Ask the user for the timeline of the movies recommended
- Using the information collected, recommend movies to the user that have a high weighted rating (according to the IMDB formula) and that satisfy the preceding conditions

The data that we have has information on the duration, genres, and timelines, but it isn't currently in a form that is directly usable. In other words, our data needs to be wrangled before it can be put to use to build this recommender.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Hr2iLUwnSVeW" executionInfo={"status": "ok", "timestamp": 1615184034200, "user_tz": -330, "elapsed": 2162, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6007342d-25e8-443f-e32b-b36c7d164cae"
#Only keep those features that we require 
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

df.head()
```

<!-- #region id="xbYQ9hTjTJv1" -->
Next, let us extract the year of release from our release_date feature:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="9zpoADQWTI0X" executionInfo={"status": "ok", "timestamp": 1615184037421, "user_tz": -330, "elapsed": 3496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b50e074e-0534-4416-bb9f-40a231941b5e"
#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Extract year from the datetime
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0

#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)

#Drop the release_date column
df = df.drop('release_date', axis=1)

#Display the dataframe
df.head()
```

<!-- #region id="9RrttzUBT5YC" -->
Upon preliminary inspection, we can observe that the genres are in a format that looks like a JSON object (or a Python dictionary). Let us take a look at the genres object of one of our movies:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="iW6RLaGzTcLQ" executionInfo={"status": "ok", "timestamp": 1615181364989, "user_tz": -330, "elapsed": 1524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f2c7b0a-edef-46ef-e7e5-b7028ff4e75b"
#Print genres of the first movie
df.iloc[0]['genres']
```

<!-- #region id="Sw0oTk1qUIzm" -->
We can observe that the output is a stringified dictionary. In order for this feature to be usable, it is important that we convert this string into a native Python dictionary. Fortunately, Python gives us access to a function called literal_eval (available in the ast library) which does exactly that. literal_eval parses any string passed into it and converts it into its corresponding Python object.

Also, each dictionary represents a genre and has two keys: id and name. However, for this exercise (as well as all subsequent exercises), we only require the name. Therefore, we shall convert our list of dictionaries into a list of strings, where each string is a genre name:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="U2SEq-hOT7hs" executionInfo={"status": "ok", "timestamp": 1615184040946, "user_tz": -330, "elapsed": 2717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="16a3d1a6-cb4e-40e3-aa12-3fe84f10ded9"
#Import the literal_eval function from ast
from ast import literal_eval

#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')

#Apply literal_eval to convert to the list object
df['genres'] = df['genres'].apply(literal_eval)

#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

df.head()
```

<!-- #region id="E8QwjwmXUli-" -->
The last step is to explode the genres column. In other words, if a particular movie has multiple genres, we will create multiple copies of the movie, with each movie having one of the genres.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 258} id="uJI5rFFNUcp5" executionInfo={"status": "ok", "timestamp": 1615181571003, "user_tz": -330, "elapsed": 19230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f4971372-1ca1-4936-8783-8a3bf57be271"
#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
gen_df.head()
```

<!-- #region id="M5HuOPiiU6_z" -->
build_chart function
1. Get user input on their preferences
2. Extract all movies that match the conditions set by the user
3. Calculate the values of m and C for only these movies and proceed to build the chart
<!-- #endregion -->

```python id="Yxr73mN7UpgU"
def build_chart(gen_df, percentile=0.8):
    #Ask for preferred genres
    print("Input preferred genre")
    genre = input()
    
    #Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())
    
    #Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())
    
    #Ask for lower limit of timeline
    print("Input earliest year")
    low_year = int(input())
    
    #Ask for upper limit of timeline
    print("Input latest year")
    high_year = int(input())
    
    #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()
    
    #Filter based on the condition
    movies = movies[(movies['genre'] == genre) & 
                    (movies['runtime'] >= low_time) & 
                    (movies['runtime'] <= high_time) & 
                    (movies['year'] >= low_year) & 
                    (movies['year'] <= high_year)]
    
    #Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)
    
    #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    
    #Calculate score using the IMDB formula
    q_movies['score'] = q_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) 
                                       + (m/(m+x['vote_count']) * C)
                                       ,axis=1)

    #Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)
    
    return q_movies
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} id="BlXFb-esVdWW" executionInfo={"status": "ok", "timestamp": 1615182062884, "user_tz": -330, "elapsed": 25098, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="83fde3a7-4420-4f14-a8fd-126c04d839fe"
#Generate the chart for top animation movies and display top 5.
build_chart(gen_df).head()
```

<!-- #region id="hSbFtc-UWuLz" -->
We can see that the movies that it outputs satisfy all the conditions we passed in as input. Since we applied IMDB's metric, we can also observe that our movies are very highly rated and popular at the same time.
<!-- #endregion -->

<!-- #region id="nOgpWczTYyP0" -->
## Content-based Recommender
<!-- #endregion -->

<!-- #region id="XO5_1g_ZYZ4C" -->
The simple recommender did not take into consideration an individual user's preferences. The knowledge-based recommender did take account of the user's preference for genres, timelines, and duration, but the model and its recommendations still remained very generic. Imagine that Alice likes the movies The Dark Knight, Iron Man, and Man of Steel. It is pretty evident that Alice has a taste for superhero movies. However, our models would not be able to capture this detail. The best it could do is suggest action movies (by making Alice input action as the preferred genre), which is a superset of superhero movies.

It is also possible that two movies have the same genre, timeline, and duration characteristics, but differ hugely in their audience. Consider The Hangover and Forgetting Sarah Marshall, for example. Both these movies were released in the first decade of the 21st century, both lasted around two hours, and both were comedies. However, the kind of audience that enjoyed these movies was very different.

We are going to build two types of content-based recommender:
1. Plot description-based recommender: This model compares the descriptions and taglines of different movies, and provides recommendations that have the most similar plot descriptions.
2. Metadata-based recommender: This model takes a host of features, such as genres, keywords, cast, and crew, into consideration and provides recommendations that are the most similar with respect to the aforementioned features.
<!-- #endregion -->

<!-- #region id="Ad_RgpyebIE5" -->
### Plot description-based recommender
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ETEGVHJxVgye" executionInfo={"status": "ok", "timestamp": 1615185474482, "user_tz": -330, "elapsed": 2285, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="01bf97a3-e8d7-4b1d-b22b-5b04a3642c09"
#Import the original file
orig_df = pd.read_csv('movies_metadata.csv', low_memory=False)

#Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6X_DTcQ9bUk6" executionInfo={"status": "ok", "timestamp": 1615185492775, "user_tz": -330, "elapsed": 3780, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0529e893-5390-4035-aa2f-8d28d378e93f"
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix
```

<!-- #region id="POsQgYaqb9jp" -->
We see that the vectorizer has created a 75,827-dimensional vector for the overview of every movie.

The next step is to calculate the pairwise cosine similarity score of every movie. In other words, we are going to create a 45,466 × 45,466 matrix, where the cell in the ith row and jth column represents the similarity score between movies i and j. We can easily see that this matrix is symmetric in nature and every element in the diagonal is 1, since it is the similarity score of the movie with itself.
<!-- #endregion -->

```python id="AxfMSGT9b6oh"
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel
```

```python id="s9Win0W6cQMV"
#Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
```

```python id="qPVKFJree0ng"
# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, tfidf_matrix=tfidf_matrix, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
```

```python colab={"base_uri": "https://localhost:8080/"} id="mTwVEbwZfvFM" executionInfo={"status": "ok", "timestamp": 1615184498614, "user_tz": -330, "elapsed": 1204, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea31d037-ca6d-40d2-d265-9728c794fa44"
#Get recommendations for The Lion King
content_recommender('The Lion King')
```

<!-- #region id="Fp004B1vgHlp" -->
We see that our recommender has suggested all of The Lion King's sequels in its top-10 list. We also notice that most of the movies in the list have to do with lions.

It goes without saying that a person who loves The Lion King is very likely to have a thing for Disney movies. They may also prefer to watch animated movies. Unfortunately, our plot description recommender isn't able to capture all this information.

Therefore, in the next section, we will build a recommender that uses more advanced metadata, such as genres, cast, crew, and keywords (or sub-genres). This recommender will be able to do a much better job of identifying an individual's taste for a particular director, actor, sub-genre, and so on.
<!-- #endregion -->

<!-- #region id="qqxbOwNygMwg" -->
### Metadata-based recommender

To build this model, we will be using the following metdata:
- The genre of the movie. 
- The director of the movie. This person is part of the crew.
- The movie's three major stars. They are part of the cast.
- Sub-genres or keywords.
<!-- #endregion -->

```python id="NPxTvHHIf4pu"
# Load the keywords and credits files
cred_df = pd.read_csv('credits.csv')
key_df = pd.read_csv('keywords.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="xworY12agg2I" executionInfo={"status": "ok", "timestamp": 1615185502919, "user_tz": -330, "elapsed": 2475, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a6f79693-b74c-486f-f9f5-3ff8a13ba0a9"
#Print the head of the credit dataframe
cred_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="iPNMzp0egk7F" executionInfo={"status": "ok", "timestamp": 1615185503745, "user_tz": -330, "elapsed": 1701, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bbf4ab20-d1d0-4da5-8de1-2b79025cbd81"
#Print the head of the keywords dataframe
key_df.head()
```

<!-- #region id="37IKPlNrg1Fr" -->
We can see that the cast, crew, and the keywords are in the familiar list of dictionaries form. Just like genres, we have to reduce them to a string or a list of strings.

Before we do this, however, we will join the three DataFrames so that all our features are in a single DataFrame. Joining pandas DataFrames is identical to joining tables in SQL. The key we're going to use to join the DataFrames is the id feature. However, in order to use this, we first need to explicitly convert is listed as an ID. This is clearly bad data. 
<!-- #endregion -->

```python id="qW-AL0ZSgmfk"
#Convert the IDs of df into int
df['id'] = df['id'].astype('float').astype('int')
```

<!-- #region id="Gk1nwgfniK6C" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 612} id="L0BOPHwug5qo" executionInfo={"status": "ok", "timestamp": 1615185515654, "user_tz": -330, "elapsed": 1435, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a21d3e94-073e-4980-ef9f-dfb616b1c6f6"
# Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

#Clean the ids of df
df['id'] = df['id'].apply(clean_ids)

#Filter all rows that have a null ID
df = df[df['id'].notnull()]

# Convert IDs into integer
df['id'] = df['id'].astype('int')
key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')

#Display the head of df
df.head()
```

```python id="mJoIFJWLicsb"
# Convert the stringified objects into the native python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ay_LiRQsirxO" executionInfo={"status": "ok", "timestamp": 1615185568670, "user_tz": -330, "elapsed": 38675, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2f7786a-79cf-4308-8f17-988a612ba0f7"
#Print the first cast member of the first movie in df
df.iloc[0]['crew'][0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="xI_DMHyLiuqG" executionInfo={"status": "ok", "timestamp": 1615185568672, "user_tz": -330, "elapsed": 35568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d4873d7-d526-485a-a82c-ca2ab577b0e6"
# Extract the director's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

#Define the new director feature
df['director'] = df['crew'].apply(get_director)

#Print the directors of the first five movies
df['director'].head()
```

<!-- #region id="qgFDCE43k1Ew" -->
Both keywords and cast are dictionary lists as well. And, in both cases, we need to extract the top three name attributes of each list. Therefore, we can write a single function to wrangle both these features. Also, just like keywords and cast, we will only consider the top three genres for every movie:
<!-- #endregion -->

```python id="5G_e9y06k16a"
# Returns the list top 3 elements or entire list; whichever is more.
def generate_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="yUdWZ6AMk4Nh" executionInfo={"status": "ok", "timestamp": 1615185831214, "user_tz": -330, "elapsed": 2104, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0499478b-64b5-4d94-c9d5-8cf93da0bd9f"
#Apply the generate_list function to cast and keywords
df['cast'] = df['cast'].apply(generate_list)
df['keywords'] = df['keywords'].apply(generate_list)

#Only consider a maximum of 3 genres
df['genres'] = df['genres'].apply(lambda x: x[:3])

# Print the new features of the first 5 movies along with title
df[['title', 'cast', 'director', 'keywords', 'genres']].head()
```

<!-- #region id="XayhoiKKlG-G" -->
In the subsequent steps, we are going to use a vectorizer to build document vectors. If two actors had the same first name (say, Ryan Reynolds and Ryan Gosling), the vectorizer will treat both Ryans as the same, although they are clearly different entities. This will impact the quality of the recommendations we receive. If a person likes Ryan Reynolds' movies, it doesn't imply that they like movies by all Ryans. 

Therefore, the last step is to strip the spaces between keywords, and actor and director names, and convert them all into lowercase. Therefore, the two Ryans in the preceding example will become ryangosling and ryanreynolds, and our vectorizer will now be able to distinguish between them:
<!-- #endregion -->

```python id="HU01ebJok9xP"
# Function to sanitize data to prevent ambiguity. It removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```

```python id="Abx4KawjlLo2"
#Apply the generate_list function to cast, keywords, director and genres
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)
```

<!-- #region id="6hqBSF-UlaPV" -->
In the plot description-based recommender, we worked with a single overview feature, which was a body of text. Therefore, we were able to apply our vectorizer directly.

However, this is not the case with our metadata-based recommender. We have four features to work with, of which three are lists and one is a string. What we need to do is create a soup that contains the actors, director, keywords, and genres. This way, we can feed this soup into our vectorizer and perform similar follow-up steps to before:
<!-- #endregion -->

```python id="JQtVIRQ3lNcY"
#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="0JznVqnvlReA" executionInfo={"status": "ok", "timestamp": 1615185922097, "user_tz": -330, "elapsed": 2403, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd52c2e8-bd37-4f06-d7c7-2295d4e9fac3"
# Create the new soup feature
df['soup'] = df.apply(create_soup, axis=1)

#Display the soup of the first movie
df.iloc[0]['soup']
```

<!-- #region id="-HgY8GKeltfz" -->
The next steps are almost identical to the corresponding steps from the previous section.

Instead of using TF-IDFVectorizer, we will be using CountVectorizer. This is because using TF-IDFVectorizer will accord less weight to actors and directors who have acted and directed in a relatively larger number of movies.

This is not desirable, as we do not want to penalize artists for directing or appearing in more movies:
<!-- #endregion -->

```python id="avhMnI9slT4F"
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

# Reset index of your df and construct reverse mapping again
df = df.reset_index()
indices2 = pd.Series(df.index, index=df['title'])
```

```python id="3_rl43IzmNpg"
# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, matrix=tfidf_matrix, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(linear_kernel(matrix[idx], matrix).flatten()))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
```

```python colab={"base_uri": "https://localhost:8080/"} id="oPB3tyWtnAO9" executionInfo={"status": "ok", "timestamp": 1615186380693, "user_tz": -330, "elapsed": 1439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1d150cd3-284f-4b94-9345-af31b395bbaa"
content_recommender('The Lion King', count_matrix, df, indices2)
```

<!-- #region id="AKZxqFjvqYvW" -->
## EDA

https://www.kaggle.com/rounakbanik/the-story-of-film
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 847} id="N9GbK0smshAd" executionInfo={"status": "ok", "timestamp": 1615187824827, "user_tz": -330, "elapsed": 1718, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15f46021-23b4-44f0-c46a-055351b3ba4a"
df = pd.read_csv('movies_metadata.csv')
df.head(2).transpose()
```

<!-- #region id="850W_dIosUK6" -->
### **Features**

- **adult:** Indicates if the movie is X-Rated or Adult.
- **belongs_to_collection:** A stringified dictionary that gives information on the movie series the particular film belongs to.
- **budget:** The budget of the movie in dollars.
- **genres:** A stringified list of dictionaries that list out all the genres associated with the movie.
- **homepage:** The Official Homepage of the move.
- **id:** The ID of the move.
- **imdb_id:** The IMDB ID of the movie.
- **original_language:** The language in which the movie was originally shot in.
- **original_title:** The original title of the movie.
- **overview:** A brief blurb of the movie.
- **popularity:** The Popularity Score assigned by TMDB.
- **poster_path:** The URL of the poster image.
- **production_companies:** A stringified list of production companies involved with the making of the movie.
- **production_countries:** A stringified list of countries where the movie was shot/produced in.
- **release_date:** Theatrical Release Date of the movie.
- **revenue:** The total revenue of the movie in dollars.
- **runtime:** The runtime of the movie in minutes.
- **spoken_languages:** A stringified list of spoken languages in the film.
- **status:** The status of the movie (Released, To Be Released, Announced, etc.)
- **tagline:** The tagline of the movie.
- **title:** The Official Title of the movie.
- **video:** Indicates if there is a video present of the movie with TMDB.
- **vote_average:** The average rating of the movie.
- **vote_count:** The number of votes by users, as counted by TMDB.
<!-- #endregion -->

<!-- #region id="lq9zGh1Go-O0" -->
References
- [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset)
- [eBook](https://learning.oreilly.com/library/view/hands-on-recommendation-systems/9781788993753/)
<!-- #endregion -->

```python id="2GOtKWhXnEFQ"

```
