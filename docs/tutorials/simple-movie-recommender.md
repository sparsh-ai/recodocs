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
    language: python
    name: python3
---

<!-- #region id="m_NQm5KcOMLz" -->
# Simple movie recommender in implicit, explicit, and cold-start settings
> Applying data cleaning, exploration, explicit KNN model, implicit ALS model, and cold-start scenario on movielens dataset

- toc: true
- badges: true
- comments: true
- categories: [Movie, Implicit, KNN]
- author: "<a href='https://github.com/topspinj/recommender-tutorial'>Jill Cates</a>"
- image:
<!-- #endregion -->

<!-- #region id="yzwCHncOE8uf" -->
## Import the Dependencies
<!-- #endregion -->

```python id="ChD__RSUNauT"
!pip install -q fuzzywuzzy python-Levenshtein implicit
```

```python id="9Js8DW0HE8ug"
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import re 
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
import implicit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

<!-- #region id="sXE4BARrE8uh" -->
## Load the Data

Let's download a small version of the [MovieLens](https://www.wikiwand.com/en/MovieLens) dataset. You can access it via the zip file url [here](https://grouplens.org/datasets/movielens/), or directly download [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip). We're working with data in `ml-latest-small.zip` and will need to add the following files to our local directory: 
- ratings.csv
- movies.csv

These are also located in the data folder inside this GitHub repository. 

Alternatively, you can access the data here: 
- https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv
- https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv

Let's load in our data and take a peek at the structure.
<!-- #endregion -->

```python id="TPH1aXhhE8ui" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="6f628bcf-f4d5-4d37-dc2c-ce93840df69c"
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
ratings.head()
```

```python id="FSIDTQsfE8uk" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="c0dbdf10-fe5c-4d2f-a4d4-1ea14d8228be"
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
movies.head()
```

<!-- #region id="1AB0Y3FaF1dX" -->
## Data Cleaning and Exploration
<!-- #endregion -->

<!-- #region id="gAnqa_yWF1dY" -->
### Converting Genres from String Format to List 

The genres column is currently a string separated with pipes. Let's convert this into a list using the "split" function.

We want 
`"Adventure|Children|Fantasy"`
to convert to this:
`[Adventure, Children, Fantasy]`.
<!-- #endregion -->

```python id="b3FbtK2dF1da" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="ad93034f-6c56-4505-84f7-c0eb45e3e0da"
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
movies.head()
```

<!-- #region id="ZiX7_ztSF1db" -->
### How many movie genres are there?

We can use Python's Counter to create a dictionary containing frequency counts of each genre in our dataset.
<!-- #endregion -->

```python id="kXnM7XXHF1db" colab={"base_uri": "https://localhost:8080/"} outputId="e5247427-3f6f-41b5-ebd7-c4e05f7c2369"
genres_counts = Counter(g for genres in movies['genres'] for g in genres)
print(f"There are {len(genres_counts)} genre labels.")
genres_counts
```

<!-- #region id="ZJjes2VqF1dc" -->
There are 20 genre labels and 19 genres that are used to describe movies in this dataset. Some movies don't have any genres, hence the label `(no genres listed)`. 

Let's remove all movies having `(no genres listed)` as its genre label. We'll also remove this from our `genre_counts` dictionary. 
<!-- #endregion -->

```python id="wtaP--SRF1dd"
movies = movies[movies['genres']!='(no genres listed)']

del genres_counts['(no genres listed)']
```

<!-- #region id="QoqCUjZ6F1dd" -->
### What are the most popular genres?

We can use `Counter`'s [most_common()](https://docs.python.org/2/library/collections.html#collections.Counter.most_common) method to get the genres with the highest movie counts.
<!-- #endregion -->

```python id="n56wR-7XF1dd" colab={"base_uri": "https://localhost:8080/"} outputId="cb2c0fe1-5962-4336-f0dd-0a1e92389395"
print("The 5 most common genres: \n", genres_counts.most_common(5))
```

<!-- #region id="DkB5GkZIF1de" -->
The top 5 genres are: `Drama`, `Comedy`, `Thriller`, `Action` and `Romance`. 

Let's also visualize genres popularity with a barplot.
<!-- #endregion -->

```python id="U6_IrqtKF1de" colab={"base_uri": "https://localhost:8080/", "height": 391} outputId="634f2cc4-363b-4da8-f189-645d7ec75401"
genres_counts_df = pd.DataFrame([genres_counts]).T.reset_index()
genres_counts_df.columns = ['genres', 'count']
genres_counts_df = genres_counts_df.sort_values(by='count', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='genres', y='count', data=genres_counts_df, palette='viridis')
plt.xticks(rotation=90)
plt.show()
```

<!-- #region id="exnlL95vF1df" -->
The plot above shows that `Drama` and `Comedy` are the two most popular movie genres. The least popular movie genres are `Westerns`, `IMAX`, and `Film-Noir`.
<!-- #endregion -->

<!-- #region id="2pNlk9T6F1df" -->
### Parsing out year from movie title

In our dataset, movie titles currently the year of release appended to it in brackets, e.g., `"Toy Story (1995)"`. We want to use the year of a movie's release as a feature, so let's parse it out from the title string and create a new `year` column for it.

We can start with writing a function that parses out year from the title string. In the code below, `extract_year_from_title()` takes in the title and does the following:

- generates a list by splitting out each word by spaces (e.g., `["Toy", "Story", "(1995)"]`)
- gets the last element of the list (e.g., `"(1995)"`)
- if the last element has brackets surrounding it, these `()` brackets get stripped (e.g., `"1995"`)
- converts the year into an integer 
<!-- #endregion -->

```python id="3gNjFuIhF1dg"
def extract_year_from_title(title):
    t = title.split(' ')
    year = None
    if re.search(r'\(\d+\)', t[-1]):
        year = t[-1].strip('()')
        year = int(year)
    return year
```

<!-- #region id="2CEO6Ka0F1dg" -->
We can test out this function with our example of `"Toy Story (1995)"`:
<!-- #endregion -->

```python id="SdsQplVCF1dh" colab={"base_uri": "https://localhost:8080/"} outputId="ff78137f-4b87-4868-a8b0-bef5c6b6bebb"
title = "Toy Story (1995)"
year = extract_year_from_title(title)
print(f"Year of release: {year}")
print(type(year))
```

<!-- #region id="hM2tjQCBF1di" -->
Our function `extract_year_from_title()` works! It's able to successfully parse out year from the title string as shown above. We can now apply this to all titles in our `movies` dataframe using Pandas' [apply()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) method.
<!-- #endregion -->

```python id="-JGlXtQfF1di" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="22904da9-4d14-45e7-8f77-00cb321ee2e3"
movies['year'] = movies['title'].apply(extract_year_from_title)
movies.head()
```

<!-- #region id="0D715fW-F1di" -->
### How many different years of release are covered in our dataset?
<!-- #endregion -->

```python id="yw8qQ5G6F1dj" colab={"base_uri": "https://localhost:8080/"} outputId="a7913242-3ddc-4259-d83a-a882c7dbcedb"
movies['year'].nunique()
```

<!-- #region id="tsC-849xF1dj" -->
There are over 100 years of release in our dataset. Let's collapse this down into decades to get a general sense of when movies were released in our dataset. 
<!-- #endregion -->

<!-- #region id="QoVnzmIhF1dj" -->
### What was the most popular decade of movie release?

Before we begin, we'll remove all movies with null year.
<!-- #endregion -->

```python id="yLNEzHugF1dk" colab={"base_uri": "https://localhost:8080/"} outputId="cc1c6014-cfec-4862-c5dc-1c4b92a8eeaf"
print(f"Original number of movies: {movies['movieId'].nunique()}")
```

```python id="Dc7pSzBmF1dk" colab={"base_uri": "https://localhost:8080/"} outputId="59a964ce-7e15-4ace-f703-5a55edc6cf95"
movies = movies[~movies['year'].isnull()]
print(f"Number of movies after removing null years: {movies['movieId'].nunique()}")
```

<!-- #region id="sWIHktG6F1dk" -->
We filtered out 24 movies that don't have a year of release. 

Now, there are two ways to get the decade of a year:

1. converting year to string, replacing the fourth (last) number with a 0
2. rounding year down to the nearest 10 

We'll show both implementations in the code below:
<!-- #endregion -->

```python id="3KNsd3QrF1dl" colab={"base_uri": "https://localhost:8080/"} outputId="6b4206a2-d6de-48bf-cb26-b8b7b7f53e5d"
x = 1995

def get_decade(year):
    year = str(year)
    decade_prefix = year[0:3] # get first 3 digits of year
    decade = f'{decade_prefix}0' # append 0 at the end
    return int(decade)

get_decade(x)
```

```python id="EYoQONVAF1dl" colab={"base_uri": "https://localhost:8080/"} outputId="93fa3006-0f10-4b32-f54f-63ef63ed1382"
def round_down(year):
    return year - (year%10)

round_down(x)
```

<!-- #region id="u6sa9iK1F1dl" -->
The two functions `get_decade()` and `round_down()` both accomplish the same thing: they both get the decade of a year.

We can apply either of these functions to all years in our `movies` dataset. We'll use `round_down()` in this example to a create a new column called `'decade'`:
<!-- #endregion -->

```python id="nVXsZjkXF1dm"
movies['decade'] = movies['year'].apply(round_down)
```

```python id="QLlkbHF1F1dm" colab={"base_uri": "https://localhost:8080/", "height": 447} outputId="c1a6fe07-3735-4bca-8167-a6c767d7f62b"
plt.figure(figsize=(10,6))
sns.countplot(movies['decade'], palette='Blues')
plt.xticks(rotation=90)
```

<!-- #region id="WYr5NlqiF1dm" -->
As we can see from the plot above, the most common decade is the 2000s followed by the 1990s for movies in our dataset.
<!-- #endregion -->

```python id="Cmx7ux-gE8um" colab={"base_uri": "https://localhost:8080/"} outputId="58139af5-8b62-4532-ddce-d7d3e5526225"
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")
```

<!-- #region id="qcXuHxW3E8um" -->
Now, let's take a look at users' rating counts. We can do this using pandas' `groupby()` and `count()` which groups the data by `userId`'s and counts the number of ratings for each userId. 
<!-- #endregion -->

```python id="UtEkdmKbE8un" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="81a400e5-01b9-4c93-bf1b-6056a1c74a2a"
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
user_freq.head()
```

```python id="Z4eY4A43E8un" colab={"base_uri": "https://localhost:8080/"} outputId="f19a1f44-0245-445e-af48-33c3dda2e706"
print(f"Mean number of ratings for a given user: {user_freq['n_ratings'].mean():.2f}.")
```

<!-- #region id="dDe7HDYtE8uo" -->
On average, a user will have rated ~165 movies. Looks like we have some avid movie watchers in our dataset.
<!-- #endregion -->

```python id="N_CA6X8dE8uo" colab={"base_uri": "https://localhost:8080/", "height": 350} outputId="46f9dc7f-5797-4d52-80fc-95c437d862f7"
sns.set_style("whitegrid")
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
ax = sns.countplot(x="rating", data=ratings, palette="viridis")
plt.title("Distribution of movie ratings")

plt.subplot(1,2,2)
ax = sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False)
plt.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
plt.xlabel("# ratings per user")
plt.ylabel("density")
plt.title("Number of movies rated per user")
plt.show()
```

<!-- #region id="CJxvIxOkE8up" -->
The most common rating is 4.0, while lower ratings such as 0.5 or 1.0 are much more rare. 
<!-- #endregion -->

<!-- #region id="IgqwQQUzE8up" -->
### Which movie has the lowest and highest average rating?
<!-- #endregion -->

```python id="Q1N70i7wE8uq" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="3171e256-8af4-439a-c9f1-7750e20b6df6"
mean_rating = ratings.groupby('movieId')[['rating']].mean()

lowest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == lowest_rated]
```

<!-- #region id="wCzgGjDfE8uq" -->
Santa with Muscles is the worst rated movie!
<!-- #endregion -->

```python id="GkSZ2o0pE8uq" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="59a04fb6-0784-4a88-ba6e-d13330acd9b3"
highest_rated = mean_rating['rating'].idxmax()
movies.loc[movies['movieId'] == highest_rated]
```

<!-- #region id="nHQzK0hKE8ur" -->
Lamerica may be the "highest" rated movie, but how many ratings does it have?
<!-- #endregion -->

```python id="vXNyFd9mE8ur" colab={"base_uri": "https://localhost:8080/", "height": 111} outputId="e53a3a8d-83dd-4e29-8b31-6c76788eac6a"
ratings[ratings['movieId']==highest_rated]
```

<!-- #region id="9yht1-MSE8us" -->
Lamerica has only 2 ratings. A better approach for evaluating movie popularity is to look at the [Bayesian average](https://en.wikipedia.org/wiki/Bayesian_average).
<!-- #endregion -->

<!-- #region id="CJ4dp2fyE8ut" -->
### Bayesian Average

Bayesian Average is defined as:

$r_{i} = \frac{C \times m + \Sigma{\text{reviews}}}{C+N}$

where $C$ represents our confidence, $m$ represents our prior, and $N$ is the total number of reviews for movie $i$. In this case, our prior will be the average rating across all movies. By defintion, C represents "the typical dataset size". Let's make $C$ be the average number of ratings for a given movie.
<!-- #endregion -->

```python id="ytMWg-HbE8uu"
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()
```

```python id="e1Z5hBcCE8uv"
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')
```

```python id="r2hkOiW-E8uv" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="080412c1-3ebb-44cf-936e-a2c023c945ea"
movie_stats = movie_stats.merge(movies[['movieId', 'title']])
movie_stats.sort_values('bayesian_avg', ascending=False).head()
```

<!-- #region id="ioewti2RE8uw" -->
Using the Bayesian average, we see that `Shawshank Redemption`, `The Godfather`, and `Fight Club` are the most highly rated movies. This result makes much more sense since these movies are critically acclaimed films.

Now which movies are the worst rated, according to the Bayesian average?
<!-- #endregion -->

```python id="ZEN4WtQ_E8ux" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="8c193e0a-05ea-4918-c8ea-02108493c76c"
movie_stats.sort_values('bayesian_avg', ascending=True).head()
```

<!-- #region id="29CoIE31E8ux" -->
With Bayesian averaging, it looks like `Speed 2: Cruise Control`, `Battlefield Earth`, and `Godzilla` are the worst rated movies. `Gypsy` isn't so bad after all!
<!-- #endregion -->

<!-- #region id="ucmiRmpaGph8" -->
## Building an Item-Item Recommender

If you use Netflix, you will notice that there is a section titled "Because you watched Movie X", which provides recommendations for movies based on a recent movie that you've watched. This is a classic example of an item-item recommendation. We will generate item-item recommendations using a technique called [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering). Let's get started! 
<!-- #endregion -->

<!-- #region id="sQ5W_oCTE8uy" -->
### Transforming the data

We will be using a technique called [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) to generate user recommendations. This technique is based on the assumption of "homophily" - similar users like similar things. Collaborative filtering is a type of unsupervised learning that makes predictions about the interests of a user by learning from the interests of a larger population.

The first step of collaborative filtering is to transform our data into a `user-item matrix` - also known as a "utility" matrix. In this matrix, rows represent users and columns represent items. The beauty of collaborative filtering is that it doesn't require any information about the users or items to generate recommendations. 
<!-- #endregion -->

<!-- #region id="zCZsygWQMOGp" -->
![image.png](data:image/png;base64,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/T2mjAWLEIhIXrAIgYjkBYsQiEhehEX9nqpJHEfmixR289nruYXAWlpofvCDH6TnPve5aeONN54o8D7gAQ8YfS7vnXfe2UJO07WUO6XP9lA21tLnwfhS+s53vpOe//znpy233HJina7Wm9122y2dd955CroSZcUEmqx99TYr1vGCn8j+HhPEgkUIRCQvWIRARPKCRQhEJC/Con5P1SSOI/NFCrv57PXcQmAtLjSf//zn0/7777+kaFD9QZaPfexjLfTWdtO1mDulzvhQNtZS/Y0rpSuvvHL0P94222yzJWtzVdB9xzvekfzPN5my0gJN1r56m5Xuf1HPZ3+PmWHBIgQikhcsQiAiecEiBCKSF2FRv6dqEseR+SKF3Xz2em4hsJYXmuqdXw972MOWFBF+67d+K335y19uobg2m67l3CltxoeysZbmbjxp9Dmphx9+eNpkk02WrMV77bVXes973uMduhJl1QSarH31Nqt2IQt2Yvt7TAgLFiEQkbxgEQIRyQsWIRCRvAiL+j1VkziOzBcp7Oaz13MLgbW+0Nxyyy3pla985ZJ/9nuve90rHXvssel73/teC8211XSt505Jsz2UjbUk87U+ls985jOp+p9oG2200ZKCbvUvKj74wQ+udSLj70GgydpXb9PDJS1EF/b3mAYWLEIgInnBIgQikhcsQiAieREW9XuqJnEcmS9S2M1nr+cWAhaau7Gqz3V89rOfvaTIsNVWW6VTTjklrV+/voXq2mgqd8qZ56FsrOWIr82R3HXXXenCCy9MT37yk5cUc6v15OlPf3q6+OKL1yaOUWcRaLL21dtkucgMndrfA50FixCISF6wCIGI5AWLEIhIXoRF/Z6qSRxH5osUdvPZ67mFgIVmEuvSSy9NT3va05YUHR74wAems88+O91+++2TB6zhR3KnnMkfysZajvjaGslPf/rTdM4556RHP/rRS9bW6h27hx12WPrc5z63tlCMdiEEmqx99TYLcdE9XIT9PZBZsAiBiOQFixCISF6wCIGI5EVY1O+pmsRxZL5IYTefvZ5bCFhopmP9x3/8R3rsYx+7pAixyy67pHe+850+8zGlCZvpip4disBQNtaheLrOuwVuuOGGdNppp6UHP/jBE+tFte9suumm6cgjj0xXXHEFLgLZBJqsffU22S60547dGwY4CxYhEJG8YBECEckLFiEQ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AgTmE2iy9tXbjH8Gff//pt5zcOEiB+SAHJADckAOLJcD9XuqJvF8d3cre5TC7sp6OtsqCdR/6FapC6ftQeCOO+5IF1xwQTrkkEPSve51r6m/cFXF3z//8z9Pl19++YpckdxZEcaFOMlQNtaFwFqFi6g+PuXMM89M++67b9p4442n/vxWfwTt4IMPTm95y1vSj3/841W4CqcksPYEmqx99Tb1fU/sF1c5IAfkgByQA3JADjTPgfo9VZN4Ee5MFXYXYRZcwwYF6gvRBhtrMAiBa665Jr361a9Ou+6669QCUTXnu+++++jzOqvP7Z33S+7MK7d4xw1lY108ufmv6HOf+9zo83Af85jHLPtzuummm6Zf//VfT29605vSjTfeOH9njiRAYKpAk7Wv3qa+74mb/yLHipUckANyQA7IATlQv6dqEk+9eev5SYXdnsF1N59AfYGd7wyOWmSBSy65JD3/+c9PD3zgA6cWj6rP463+yfff/u3fpurzPNt8yZ02Wovddigb62Irzr666l31H/7wh9Pznve8tOOOO079eax+pqp33B944IHpn//5n9P1118/+6ReJUCgk0CTta/eplNnAzrY/h6TxYJFCEQkL1iEQETygkUIRCQvwqJ+T9UkjiPzRQq7+ez13ELAQtMCa8BNq6LSv//7v6fDDz88bbnlllOLSlWRd5999hn9saZvf/vbGxyt3Nkg0WAaDGVjHQzo/7vQ6667bvTRCb//+7+ftttuu6k/d9XPUfUzWX1m7pvf/Ob0ox/9aGjDdL0EBivQZO2rtxnsQFteuP09wFiwCIGI5AWLEIhIXrAIgYjkRVjU76maxHFkvkhhN5+9nlsIWGhaYBXSdP369aMCUvWuwOU+j7fKi7333nv0uZ9XXnnl1JHLnaksg3xyKBvrouPeddddqXqX/Ctf+crR/yRZ7vNyq5+d+9///umoo45K733ve9Mtt9yy6ENzfQSKFGiy9tXbFIkwZVD290BhwSIEIpIXLEIgInnBIgQikhdhUb+nahLHkfkihd189npuIWChaYFVYNMf/vCHo3/yfdBBB6Xq8zzr+VCPH/WoR6WXvOQl6eMf/3i68847RxL11wukWVNDGsrGuoiTcsMNN6R3vetd6eijj04PetCDlv0Zqn5eHvawh40+GuUjH/lIqt5F74sAgbwCTda+epu8V9tf7/b3sGbBIgQikhcsQiAiecEiBCKSF2FRv6dqEseR+SKF3Xz2em4hYKFpgVV40+qfgJ9zzjnp137t12YWebfffvt0xBFHTBSwCqcpfnhD2VgXYSL+7//+b/RZuSeeeGJ6whOekDbZZJOJn4X6mlq9I37//fdPp512WvrCF76wCJfvGggQqAk0WfvqbWqHFh3W17GiB9pgcCwCiQWLEIhIXrAIgYjkBYsQiKh+T9UkjiPzRQq7+ez13ELAotsCaw01rd6F+KY3vSkdfPDB6d73vveyhatx/lR/fO2LX/ziGhIqa6hD2VhzqFcfr/D5z38+nXHGGaM/arahn4fqDxVWH7Hwzne+M9144405LlmfBAg0FGiy9tXbNDzt4JuN9/bq+1r/YhEZwIJFCEQkL1iEQETygkUIRFS/p2oSx5H5ouLuhIYCn2/Kh9mzRXeY89bnVVef//n+978//eEf/mHaYYcdZhZ5q9ePPPLIdO6556Yf/OAHfV6mvjoIWN8DryrkVu+u/Yd/+If0zGc+c/RZuPV18p5x9Y7d6o8OVp+rW32+bnW8LwIEhiHQZO2rtxnGqLpfZX2d6362YZ+BRcwfCxYhEJG8YBECEckLFiEQUf2eqkkcR+aLFHbz2eu5hYBFtwWWpqOi1Wc/+9l08sknzyzwVnm10UYbpcc97nHpxS9+cbrooovST37yE4ILKjCUjXU1+G677bb06U9/evSO3Ood6ttuu+0Gc3vXXXdNf/InfzL6w2felbsas+KcBPoRaLL21dv0c1X5e3FvGHPAgkUIRCQvWIRARPKCRQhEJC/Con5P1SSOI/NFCrv57PXcQsBC0wJL0wmBeu5U72zcbrvtZhbEqj/OVr2z8S/+4i/ShRde6J+pT2jmfTCUjXUllG666ab0n//5n+kVr3hFOuCAA9KWW245M2+rPK/eif6c5zwnvfnNb07r1q1bictwDgIEFkCgydpXb7MAl9zLJdT39146XOBOWMTksGARAhHJCxYhEJG8YBECEdXvqZrEcWS+SGE3n72eWwhYdFtgaTohcM/cufPOO9P//M//pFNPPTU9+clPnvkH2KpjN95447Tnnnum5z//+en888/30Q0Tuv0+GMrG2lbljjvuSJdddll6wxvekI4++uj06Ec/epR39dydFj/4wQ9Ov/d7vzf6OIb//d//bdut9gQIDESgydpXbzOQYXW+zPq62PlkAz8Bi5hAFixCICJ5wSIEIpIXLEIgovo9VZM4jswXKezms9dzCwGLbgssTScENpQ769evH3027/Oe97y0++67jz6aoX7MtHinnXZKz3rWs1L1x9g+8YlPpFtvvXWiTw9WR2AoG+uGRn/VVVeld7/73aOP/9h///0bvRu3ysNHPvKRo8LvOeeck6688soNdeN1AgQKEWiy9tXbFDLsDQ6jvj9vsHHhDVjEBLNgEQIRyQsWIRCRvGARAhHV76maxHFkvkhhN5+9nlsIWHRbYGk6IdA2d370ox+l973vfelFL3pRevzjH5+qPzpVP8e0uPr4hr322isdd9xxo38GX717snoXpq+VFRjKxjoe9e23354uv/zy9KY3vSm94AUvGL1D/L73ve8G86nKsXFOHX/88em8885L11xzzfi0vhMgsMYEmqx99TZrhae+H6+VMS83ThYhw4JFCEQkL1iEQETygkUIRFS/p2oSx5H5IoXdfPZ6biFg0W2BpemEQNfc+fGPf5w++MEPphNPPDHtt99+aYsttmhUmKvaVYXhP/iDP0ive93r0sUXX5yqz031Nb/AIm+s3/3ud9OHPvShdNZZZ6VjjjlmVOjffPPNG+VKlaMPf/jDRx+r4F3g8+eHIwmUKtBk7au3KdXhnuPqur/f83xDfswiZo8FixCISF6wCIGI5AWLEIiofk/VJI4j80UKu/ns9dxCwKLbAkvTCYGVzp3bbrstXXLJJenss89ORx55ZNp1110bfXxDdR0bbbRResQjHpEOPfTQ9PKXvzy97W1vS5///OfTzTffPHHNHkwXWISN9eqrrx4V+qsCbFW0f9KTnpTud7/7NS7gVnnwcz/3c+nAAw9MJ510Urrgggt8bvP06fYsAQL/T6DJ2ldvs1bgVnp/H7Ibi5g9FixCICJ5wSIEIpIXLEIgovo9VZM4jswXKezms9dzCwGLbgssTScE+sidG2+8MV100UWjP8h28MEHp5//+Z9vVeirCr4Pe9jD0kEHHTT6I22vf/3r00c/+tG0bt26dNddd02MZy0/6GNjrf64XlW8/fCHP5ze+MY3ppe85CXpsMMOS4973OPSfe5zn9bzWhXyq+Nf9apXpQsvvHA0p2t5Do2dAIH2Ak3Wvnqb9j0M84g+9vehyLCImWLBIgQikhcsQiAiecEiBCKq31M1iePIfJHC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DsgBOVBMDnTfFp2BAAECBAgQIECAAAECBAgMQ0BhV0GnmILOcu/g87x3d8qBtZMDw9h6XSUBAgQIECBAgAABAgSWf1+iAAAIfUlEQVQIEOguoLCrsKuwKwfkgBwoJge6b4vOQIAAAQIECBAgQIAAAQIEhiFQXGF3GOyukgABAgQIECBAgAABAgQIECBAgAABAvMLKOzOb+dIAgQIECBAgAABAgQIECBAgAABAgQIZBFQ2M3CrlMCBAgQIECAAAECBAgQIECAAAECBAjML6CwO7+dIwkQIECAAAECBAgQIECAAAECBAgQIJBFQGE3C7tOCRAgQIAAAQIECBAgQIAAAQIECBAgML+Awu78do4kQIAAAQIECBAgQIAAAQIECBAgQIBAFgGF3SzsOiVAgAABAgQIECBAgAABAgQIECBAgMD8Agq789s5kgABAgQIECBAgAABAgQIECBAgAABAlkEFHazsOuUAAECBAgQIECAAAECBAgQIECAAAEC8wso7M5v50gCBAgQIECAAAECBAgQIECAAAECBAhkEVDYzcKuUwIECBAgQIAAAQIECBAgQIAAAQIECMwvoLA7v50jCRAgQIAAAQIECBAgQIAAAQIECBAgkEVAYTcLu04JECBAgAABAgQIECBAgAABAgQIECAwv4DC7vx2jiRAgAABAgQIECBAgAABAgQIECBAgEAWAYXdLOw6JUCAAAECBAgQIECAAAECBAgQIECAwPwCCrvz2zmSAAECBAgQIECAAAECBAgQIECAAAECWQQUdrOw65QAAQIECBAgQIAAAQIECBAgQIAAAQLzCyjszm/nSAIECBAgQIAAAQIECBAgQIAAAQIECGQRUNjNwq5TAgQIECBAgAABAgQIECBAgAABAgQIzC+gsDu/nSMJECBAgAABAgQIECBAgAABAgQIECCQRUBhNwu7TgkQIECAAAECBAgQIECAAAECBAgQIDC/gMLu/HaOJECAAAECBAgQIECAAAECBAgQIECAQBYBhd0s7DolQIAAAQIECBAgQIAAAQIECBAgQIDA/AIKu/PbOZIAAQIECBAgQIAAAQIECBAgQIAAAQJZBBR2s7DrlAABAgQIECBAgAABAgQIECBAgAABAvMLKOzOb+dIAgQIECBAgAABAgQIECBAgAABAgQIZBFQ2M3CrlMCBAgQIECAAAECBAgQIECAAAECBAjML6CwO7+dIwkQIECAAAECBAgQIECAAAECBAgQIJBFQGE3C7tOCRAgQIAAAQIECBAgQIAAAQIECBAgML+Awu78do4kQIAAAQIECBAgQIAAAQIECBAgQIBAFgGF3SzsOiVAgAABAgQIECBAgAABAgQIECBAgMD8Agq789s5kgABAgQIECBAgAABAgQIECBAgAABAlkEFHazsOuUAAECBAgQIECAAAECBAgQIECAAAEC8wso7M5v50gCBAgQIECAAAECBAgQIECAAAECBAhkEVDYzcKuUwIECBAgQIAAAQIECBAgQIAAAQIECMwvoLA7v50jCRAgQIAAAQIECBAgQIAAAQIECBAgkEVAYTcLu04JECBAgAABAgQIECBAgAABAgQIECAwv4DC7vx2jiRAgAABAgQIECBAgAABAgQIECBAgEAWAYXdLOw6JUCAAAECBAgQIECAAAECBAgQIECAwPwCCrvz2zmSAAECBAgQIECAAAECBAgQIECAAAECWQQUdrOw65QAAQIECBAgQIAAAQIECBAgQIAAAQLzCyjszm/nSAIECBAgQIAAAQIECBAgQIAAAQIECGQRUNjNwq5TAgQIECBAgAABAgQIECBAgAABAgQIzC+gsDu/nSMJECBAgAABAgQIECBAgAABAgQIECCQRUBhNwu7TgkQIECAAAECBAgQIECAAAECBAgQIDC/gMLu/HaOJECAAAECBAgQIECAAAECBAgQIECAQBYBhd0s7DolQIAAAQIECBAgQIAAAQIECBAgQIDA/AIKu/PbOZIAAQIECBAgQIAAAQIECBAgQIAAAQJZBBR2s7DrlAABAgQIECBAgAABAgQIECBAgAABAvMLKOzOb+dIAgQIECBAgAABAgQIECBAgAABAgQIZBFQ2M3CrlMCBAgQIECAAAECBAgQIECAAAECBAjML6CwO7+dIwkQIECAAAECBAgQIECAAAECBAgQIJBFQGE3C7tOCRAgQIAAAQIECBAgQIAAAQIECBAgML+Awu78do4kQIAAAQIECBAgQIAAAQIECBAgQIBAFgGF3SzsOiVAgAABAgQIECBAgAABAgQIECBAgMD8Agq789s5kgABAgQIECBAgAABAgQIECBAgAABAlkEFHazsOuUAAECBAgQIECAAAECBAgQIECAAAEC8wso7M5v50gCBAgQIECAAAECBAgQIECAAAECBAhkEVDYzcKuUwIECBAgQIAAAQIECBAgQIAAAQIECMwvoLA7v50jCRAgQIAAAQIECBAgQIAAAQIECBAgkEVAYTcLu04JECBAgAABAgQIECBAgAABAgQIECAwv4DC7vx2jiRAgAABAgQIECBAgAABAgQIECBAgEAWAYXdLOw6JUCAAAECBAgQIECAAAECBAgQIECAwPwCCrvz2zmSAAECBAgQIECAAAECBAgQIECAAAECWQQUdrOw65QAAQIECBAgQIAAAQIECBAgQIAAAQLzCyjszm/nSAIECBAgQIAAAQIECBAgQIAAAQIECGQRUNjNwq5TAgQIECBAgAABAgQIECBAgAABAgQIzC+gsDu/nSMJECBAgAABAgQIECBAgAABAgQIECCQRUBhNwu7TgkQIECAAAECBAgQIECAAAECBAgQIDC/gMLu/HaOJECAAAECBAgQIECAAAECBAgQIECAQBYBhd0s7DolQIAAAQIECBAgQIAAAQIECBAgQIDA/AIKu/PbOZIAAQIECBAgQIAAAQIECBAgQIAAAQJZBBR2s7DrlAABAgQIECBAgAABAgQIECBAgAABAvML/P9JMfbWmBTUewAAAABJRU5ErkJggg==)
<!-- #endregion -->

<!-- #region id="gjLiI7JbE8uy" -->
The `create_X()` function outputs a sparse matrix X with four mapper dictionaries:
- **user_mapper:** maps user id to user index
- **movie_mapper:** maps movie id to movie index
- **user_inv_mapper:** maps user index to user id
- **movie_inv_mapper:** maps movie index to movie id

We need these dictionaries because they map which row and column of the utility matrix corresponds to which user ID and movie ID, respectively.

The **X** (user-item) matrix is a [scipy.sparse.csr_matrix](scipylinkhere) which stores the data sparsely.
<!-- #endregion -->

```python id="A8JQWUGtE8uz"
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
```

```python id="nnAsL8MdE8uz"
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
```

<!-- #region id="Bo9HhKO0E8u2" -->
Let's check out the sparsity of our X matrix.

Here, we calculate sparsity by dividing the number of non-zero elements by total number of elements as described in the equation below: 

$$S=\frac{\text{# non-zero elements}}{\text{total elements}}$$
<!-- #endregion -->

```python id="Abdey5CwE8u3" colab={"base_uri": "https://localhost:8080/"} outputId="cc7ccab2-c940-4141-b90a-ef19a1f9cd3e"
sparsity = X.count_nonzero()/(X.shape[0]*X.shape[1])

print(f"Matrix sparsity: {round(sparsity*100,2)}%")
```

<!-- #region id="9p2q4RhKE8u4" -->
Only 1.7% of cells in our user-item matrix are populated with ratings. But don't be discouraged by this sparsity! User-item matrices are typically very sparse. A general rule of thumb is that your matrix sparsity should be no lower than 0.5% to generate decent results.
<!-- #endregion -->

<!-- #region id="RHkZvrsKE8u5" -->
**Writing your matrix to a file**

We're going to save our user-item matrix for the next part of this tutorial series. Since our matrix is represented as a scipy sparse matrix, we can use the [scipy.sparse.save_npz](https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.sparse.load_npz.html) method to write the matrix to a file. 
<!-- #endregion -->

```python id="DHBCXQPAE8u6"
save_npz('user_item_matrix.npz', X)
```

<!-- #region id="kPuRkNQUE8u6" -->
### Finding similar movies using k-Nearest Neighbours

This approach looks for the $k$ nearest neighbours of a given movie by identifying $k$ points in the dataset that are closest to movie $m$. kNN makes use of distance metrics such as:

1. Cosine similarity
2. Euclidean distance
3. Manhattan distance
4. Pearson correlation 

Although difficult to visualize, we are working in a M-dimensional space where M represents the number of movies in our X matrix. 
<!-- #endregion -->

<!-- #region id="BTgAlqHLOAIM" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

```python id="nKVcgXYKE8u7"
from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
```

<!-- #region id="kK_iTvHCE8u8" -->
`find_similar_movies()` takes in a movieId and user-item X matrix, and outputs a list of $k$ movies that are similar to the movieId of interest. 

Let's see how it works in action. We will first create another mapper that maps `movieId` to `title` so that our results are interpretable. 
<!-- #endregion -->

```python id="n53VmB8mE8u8" colab={"base_uri": "https://localhost:8080/"} outputId="839c481b-6123-4965-810b-0271e4fbc684"
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1

similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}")
for i in similar_ids:
    print(movie_titles[i])
```

<!-- #region id="uHi3vuR5E8u9" -->
The results above show the 10 most similar movies to Toy Story. Most movies in this list are family movies from the 1990s, which seems pretty reasonable. Note that these recommendations are based solely on user-item ratings. Movie features such as genres are not taken into consideration in this approach.  

You can also play around with the kNN distance metric and see what results you would get if you use "manhattan" or "euclidean" instead of "cosine".
<!-- #endregion -->

```python id="guTmgTw1E8u-" colab={"base_uri": "https://localhost:8080/"} outputId="d43996f4-fb55-4480-9516-e0ebbcaee276"
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1
similar_ids = find_similar_movies(movie_id, X, k=10, metric="euclidean")

movie_title = movie_titles[movie_id]
print(f"Because you watched {movie_title}:")
for i in similar_ids:
    print(movie_titles[i])
```

<!-- #region id="Kjtx3RhTG2d1" -->
## Handling the Cold Start Problem with Content-Based Filtering

Collaborative filtering relies solely on user-item interactions within the utility matrix. The issue with this approach is that brand new users or items with no interactions get excluded from the recommendation system. This is called the "cold start" problem. Content-based filtering is a way to handle this problem by generating recommendations based on user and item features. We will generate item-item recommendations using content-based filtering.
<!-- #endregion -->

<!-- #region id="T2-n45P7F1dn" -->
### Transforming the Data

In order to build a content-based filtering recommender, we need to set up our dataset so that rows represent movies and columns represent features (i.e., genres and decades).

First, we need to manipulate the `genres` column so that each genre is represented as a separate binary feature. "1" indicates that the movie falls under a given genre, while "0" does not. 
<!-- #endregion -->

```python id="rzfF9M0FF1dn"
genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))
```

<!-- #region id="JPhLcZhDF1do" -->
Let's take a look at what the movie genres columns look like:
<!-- #endregion -->

```python id="TVXd1_82F1do" colab={"base_uri": "https://localhost:8080/", "height": 241} outputId="5537bc12-57ff-4da4-d543-0eab203be7ec"
movies[genres].head()
```

<!-- #region id="sZrFABRXF1dp" -->
Great! Our genres columns are represented as binary feautres. The next step is to wrangle our `decade` column so that each decade has its own column. We can do this using pandas' [get_dummies()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) function, which works by creating a categorical variable into binary variables.
<!-- #endregion -->

```python id="BIBbW2KjF1dp" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="573fd243-a089-4b6b-cc26-03419f463bcb"
movie_decades = pd.get_dummies(movies['decade'])
movie_decades.head()
```

<!-- #region id="aaxkVEKYF1dq" -->
Now, let's create a new `movie_features` dataframe by combining our genres features and decade features. We can do this using pandas' [concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html) function which concatenates (appends) genres and decades into a single dataframe.
<!-- #endregion -->

```python id="Ns-cKboRF1dq" colab={"base_uri": "https://localhost:8080/", "height": 241} outputId="296fd3b2-69cb-4bc0-ca08-c1c4ff48609c"
movie_features = pd.concat([movies[genres], movie_decades], axis=1)
movie_features.head()
```

<!-- #region id="neDXHuCgF1dr" -->
Our `movie_features` dataframe is ready. The next step is to start building our recommender. 
<!-- #endregion -->

<!-- #region id="lN753p5RF1dt" -->
### Building a "Similar Movies" Recommender Using Cosine Similarity

We're going to build our item-item recommender using a similarity metric called [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). 

Cosine similarity looks at the cosine angle between two vectors (e.g., $A$ and $B$). The smaller the cosine angle, the higher the degree of similarity between $A$ and $B$. You can calculate the similarity between $A$ and $B$ with this equation:

$$\cos(\theta) = \frac{A\cdot B}{||A|| ||B||}$$

In this tutorial, we're going to use scikit-learn's cosine similarity [function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) to generate a cosine similarity matrix of shape $(n_{\text{movies}}, n_{\text{movies}})$. With this cosine similarity matrix, we'll be able to extract movies that are most similar to the movie of interest.
<!-- #endregion -->

```python id="kJ585XD-F1dt" colab={"base_uri": "https://localhost:8080/"} outputId="cb7b42d7-f9f2-4a2f-f321-cd4d4e3c1c1a"
cosine_sim = cosine_similarity(movie_features, movie_features)
print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")
```

<!-- #region id="dkPhlDO8F1du" -->
As expected, after passing the `movie_features` dataframe into the `cosine_similarity()` function, we get a cosine similarity matrix of shape $(n_{\text{movies}}, n_{\text{movies}})$.

This matrix is populated with values between 0 and 1 which represent the degree of similarity between movies along the x and y axes.
<!-- #endregion -->

<!-- #region id="YISwPL0FF1dv" -->
### Let's create a movie finder function

Let's say we want to get recommendations for movies that are similar to Jumanji. To get results from our recommender, we need to know the exact title of a movie in our dataset. 

In our dataset, Jumanji is actually listed as `'Jumanji (1995)'`. If we misspell Jumanji or forget to include its year of release, our recommender won't be able to identify which movie we're interested in.  

To make our recommender more user-friendly, we can use a Python package called [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/) which will find the most similar title to a string that you pass in. Let's create a function called `movie_finder()` which take advantage of `fuzzywuzzy`'s string matching algorithm to get the most similar title to a user-inputted string. 
<!-- #endregion -->

```python id="h2HqtO7qF1dv"
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]
```

<!-- #region id="308_MlFaF1dw" -->
Let's test this out with our Jumanji example. 
<!-- #endregion -->

```python id="SIL6w8bQF1dw" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="b09f211c-c8b7-404f-9b27-d43df321227e"
title = movie_finder('juminji')
title
```

<!-- #region id="Q-eZJqeFF1dx" -->
To get relevant recommendations for Jumanji, we need to find its index in the cosine simialrity matrix. To identify which row we should be looking at, we can create a movie index mapper which maps a movie title to the index that it represents in our matrix. 

Let's create a movie index dictionary called `movie_idx` where the keys are movie titles and values are movie indices:
<!-- #endregion -->

```python id="_ePelWunF1dx" colab={"base_uri": "https://localhost:8080/"} outputId="4bf52805-c3d8-4fec-f0a3-e801e86c83d2"
movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]
idx
```

<!-- #region id="j30w1MgLF1dy" -->
Using this handy `movie_idx` dictionary, we know that Jumanji is represented by index 1 in our matrix. Let's get the top 10 most similar movies to Jumanji.
<!-- #endregion -->

```python id="Ucv5-AawF1dy"
n_recommendations=10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
similar_movies = [i[0] for i in sim_scores]
```

<!-- #region id="TuFlPwvxF1dy" -->
`similar_movies` is an array of indices that represents Jumanji's top 10 recommendations. We can get the corresponding movie titles by either creating an inverse `movie_idx` mapper or using `iloc` on the title column of the `movies` dataframe.
<!-- #endregion -->

```python id="WFZgduiiF1dy" colab={"base_uri": "https://localhost:8080/"} outputId="8e051a05-e71b-408a-88b9-4c35eb95b3b3"
print(f"Because you watched {title}:")
movies['title'].iloc[similar_movies]
```

<!-- #region id="wgtE76QzF1dz" -->
Cool! These recommendations seem pretty relevant and similar to Jumanji. The first 5 movies are family-friendly films from the 90s. 

We can test our recommender further with other movie titles. For your convenience, I've packaged the steps into a single function which takes in the movie title of interest and number of recommendations. 
<!-- #endregion -->

```python id="q4FquYfpF1dz"
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Recommendations for {title}:")
    print(movies['title'].iloc[similar_movies])
```

```python id="idxwY6wRF1d0" colab={"base_uri": "https://localhost:8080/"} outputId="ed8acfcd-1cf4-4de6-fc02-96fd868d6e75"
get_content_based_recommendations('aladin', 5)
```

<!-- #region id="zxlGP6fJIyYE" -->
## Building a Recommender System with Implicit Feedback

In this section, we will build an implicit feedback recommender system using the [implicit](https://github.com/benfred/implicit) package.

What is implicit feedback, exactly? Let's revisit collaborative filtering. In [Part 1](https://github.com/topspinj/recommender-tutorial/blob/master/part-1-item-item-recommender.ipynb), we learned that [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) is based on the assumption that `similar users like similar things`. The user-item matrix, or "utility matrix", is the foundation of collaborative filtering. In the utility matrix, rows represent users and columns represent items. 

The cells of the matrix are populated by a given user's degree of preference towards an item, which can come in the form of:

1. **explicit feedback:** direct feedback towards an item (e.g., movie ratings which we explored in [Part 1](https://github.com/topspinj/recommender-tutorial/blob/master/part-1-item-item-recommender.ipynb))
2. **implicit feedback:** indirect behaviour towards an item (e.g., purchase history, browsing history, search behaviour)

Implicit feedback makes assumptions about a user's preference based on their actions towards items. Let's take Netflix for example. If you binge-watch a show and blaze through all seasons in a week, there's a high chance that you like that show. However, if you start watching a series and stop halfway through the first episode, there's suspicion to believe that you probably don't like that show. 
<!-- #endregion -->

```python id="eAnEN09xIyYN"
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
```

<!-- #region id="OZTKDZyVIyYP" -->
For this implicit feedback tutorial, we'll treat movie ratings as the number of times that a user watched a movie. For example, if Jane (a user in our database) gave `Batman` a rating of 1 and `Legally Blonde` a rating of 5, we'll assume that Jane watched Batman one time and Legally Blonde five times. 
<!-- #endregion -->

<!-- #region id="38uKPEcjIyYQ" -->
### Transforming the Data

We need to transform the `ratings` dataframe into a user-item matrix where rows represent users and columns represent movies. The cells of this matrix will be populated with implicit feedback: in this case, the number of times a user watched a movie. 

The `create_X()` function outputs a sparse matrix **X** with four mapper dictionaries:
- **user_mapper:** maps user id to user index
- **movie_mapper:** maps movie id to movie index
- **user_inv_mapper:** maps user index to user id
- **movie_inv_mapper:** maps movie index to movie id

We need these dictionaries because they map which row and column of the utility matrix corresponds to which user ID and movie ID, respectively.

The **X** (user-item) matrix is a [scipy.sparse.csr_matrix](scipylinkhere) which stores the data sparsely.
<!-- #endregion -->

```python id="ufs5319IIyYS"
def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
```

```python id="igP7vyw-IyYU"
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
```

<!-- #region id="t3_giHPnIyYU" -->
### Creating Movie Title Mappers

We need to interpret a movie title from its index in the user-item matrix and vice versa. Let's create 2 helper functions that make this interpretation easy:

- `get_movie_index()` - converts a movie title to movie index
    - Note that this function uses [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)'s string matching to get the approximate movie title match based on the string that gets passed in. This means that you don't need to know the exact spelling and formatting of the title to get the corresponding movie index.
- `get_movie_title()` - converts a movie index to movie title
<!-- #endregion -->

```python id="grgTTgwCIyYV"
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

movie_title_mapper = dict(zip(movies['title'], movies['movieId']))
movie_title_inv_mapper = dict(zip(movies['movieId'], movies['title']))

def get_movie_index(title):
    fuzzy_title = movie_finder(title)
    movie_id = movie_title_mapper[fuzzy_title]
    movie_idx = movie_mapper[movie_id]
    return movie_idx

def get_movie_title(movie_idx): 
    movie_id = movie_inv_mapper[movie_idx]
    title = movie_title_inv_mapper[movie_id]
    return title 
```

<!-- #region id="n5mmbIuOIyYV" -->
It's time to test it out! Let's get the movie index of `Legally Blonde`. 
<!-- #endregion -->

```python id="j-n5nHwDIyYW" colab={"base_uri": "https://localhost:8080/"} outputId="1e1cdb58-df76-438d-c597-103d12dfc42a"
get_movie_index('Legally Blonde')
```

<!-- #region id="c4Gliw9LIyYW" -->
Let's pass this index value into `get_movie_title()`. We're expecting Legally Blonde to get returned.
<!-- #endregion -->

```python id="ahUOVknvIyYX" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="9f569f04-b724-4b7a-bfb0-bf9f0a662923"
get_movie_title(3282)
```

<!-- #region id="h_fO8nlxIyYX" -->
Great! These helper functions will be useful when we want to interpret our recommender results.
<!-- #endregion -->

<!-- #region id="T-15xfBBIyYY" -->
### Building Our Implicit Feedback Recommender Model


We've transformed and prepared our data so that we can start creating our recommender model.

The [implicit](https://github.com/benfred/implicit) package is built around a linear algebra technique called [matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)), which can help us discover latent features underlying the interactions between users and movies. These latent features give a more compact representation of user tastes and item descriptions. Matrix factorization is particularly useful for very sparse data and can enhance the quality of recommendations. The algorithm works by factorizing the original user-item matrix into two factor matrices:

- user-factor matrix (n_users, k)
- item-factor matrix (k, n_items)

We are reducing the dimensions of our original matrix into "taste" dimensions. We cannot interpret what each latent feature $k$ represents. However, we could imagine that one latent feature may represent users who like romantic comedies from the 1990s, while another latent feature may represent movies which are independent foreign language films.

$$X_{mn} \approx P_{mk} \times Q_{nk}^T = \hat{X}$$

<img src="images/matrix-factorization.png" width="60%"/>

In traditional matrix factorization, such as SVD, we would attempt to solve the factorization at once which can be very computationally expensive. As a more practical alternative, we can use a technique called `Alternating Least Squares (ALS)` instead. With ALS, we solve for one factor matrix at a time:

- Step 1: hold user-factor matrix fixed and solve for the item-factor matrix
- Step 2: hold item-factor matrix fixed and solve for the user-item matrix

We alternate between Step 1 and 2 above, until the dot product of the item-factor matrix and user-item matrix is approximately equal to the original X (user-item) matrix. This approach is less computationally expensive and can be run in parallel.

The [implicit](https://github.com/benfred/implicit) package implements matrix factorization using Alternating Least Squares (see docs [here](https://implicit.readthedocs.io/en/latest/als.html)). Let's initiate the model using the `AlternatingLeastSquares` class.
<!-- #endregion -->

```python id="TWZbWWmmIyYZ"
model = implicit.als.AlternatingLeastSquares(factors=50, use_gpu=False)
```

<!-- #region id="Teg2v2iEIyYa" -->
This model comes with a couple of hyperparameters that can be tuned to generate optimal results:

- factors ($k$): number of latent factors,
- regularization ($\lambda$): prevents the model from overfitting during training

In this tutorial, we'll set $k = 50$ and $\lambda = 0.01$ (the default). In a real-world scenario, I highly recommend tuning these hyperparameters before generating recommendations to generate optimal results.

The next step is to fit our model with our user-item matrix. 
<!-- #endregion -->

```python id="A4okNfkXIyYb" colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["dc4a760deb274503a96719e55b57671b", "a089930bb0764fd0bc48b913cfa317f2", "da03231059ee43c090e03a59aaeddd61", "b45f3a45776c433fa85ce21838a22d7d", "bf973860201a4f1d841e9e08cc70f448", "0ecc80f43e5e41fd8a79383a61de3e35", "250fd11bb37a4daab20d489ef22fca0b", "935fa43fcffc454d9437141ddf219a4c"]} outputId="66b65257-6874-49fe-8110-c046ce844d0e"
model.fit(X)
```

<!-- #region id="EuOcJLGIIyYb" -->
Now, let's test out the model's recommendations. We can use the model's `similar_items()` method which returns the most relevant movies of a given movie. We can use our helpful `get_movie_index()` function to get the movie index of the movie that we're interested in.
<!-- #endregion -->

```python id="d7BnI91TIyYb" colab={"base_uri": "https://localhost:8080/"} outputId="6fa0532a-8f8f-4ae1-b237-42d54b2041c2"
movie_of_interest = 'forrest gump'

movie_index = get_movie_index(movie_of_interest)
related = model.similar_items(movie_index)
related
```

<!-- #region id="xxdtZ6ZwIyYc" -->
The output of `similar_items()` is not user-friendly. We'll need to use our `get_movie_title()` function to interpret what our results are. 
<!-- #endregion -->

```python id="QLcgZ0IgIyYc" colab={"base_uri": "https://localhost:8080/"} outputId="2c347fcd-507c-48fe-9fa2-5fa03abe6d3d"
print(f"Because you watched {movie_finder(movie_of_interest)}...")
for r in related:
    recommended_title = get_movie_title(r[0])
    if recommended_title != movie_finder(movie_of_interest):
        print(recommended_title)
```

<!-- #region id="QO0T9u_HIyYd" -->
When we treat user ratings as implicit feedback, the results look pretty good! You can test out other movies by changing the `movie_of_interest` variable.
<!-- #endregion -->

<!-- #region id="hINJBrH9IyYd" -->
### Generating User-Item Recommendations

A cool feature of [implicit](https://github.com/benfred/implicit) is that you can pull personalized recommendations for a given user. Let's test it out on a user in our dataset.
<!-- #endregion -->

```python id="7EaolVg7IyYd"
user_id = 95
```

```python id="NrSaS7hZIyYe" colab={"base_uri": "https://localhost:8080/"} outputId="9009e838-d7f8-4f96-a37b-7f249a06a6ef"
user_ratings = ratings[ratings['userId']==user_id].merge(movies[['movieId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
print(f"Number of movies rated by user {user_id}: {user_ratings['movieId'].nunique()}")
```

<!-- #region id="xi3RLzBAIyYe" -->
User 95 watched 168 movies. Their highest rated movies are below:
<!-- #endregion -->

```python id="dvN7zjeRIyYf" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="1f5bed0e-d8df-44c3-b0fc-7273a47e6ec2"
user_ratings = ratings[ratings['userId']==user_id].merge(movies[['movieId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
top_5 = user_ratings.head()
top_5
```

<!-- #region id="qK7RYCp5IyYf" -->
Their lowest rated movies:
<!-- #endregion -->

```python id="ygiaO33-IyYg" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="34149ee4-fe1c-41bd-a6dc-075db1eaf8aa"
bottom_5 = user_ratings[user_ratings['rating']<3].tail()
bottom_5
```

<!-- #region id="xxwj5F2jIyYg" -->
Based on their preferences above, we can get a sense that user 95 likes action and crime movies from the early 1990's over light-hearted American comedies from the early 2000's. Let's see what recommendations our model will generate for user 95.

We'll use the `recommend()` method, which takes in the user index of interest and transposed user-item matrix. 
<!-- #endregion -->

```python id="Id54l7ooIyYg" colab={"base_uri": "https://localhost:8080/"} outputId="19401cd3-f9b9-400a-f9f8-cdb0eb22384b"
X_t = X.T.tocsr()

user_idx = user_mapper[user_id]
recommendations = model.recommend(user_idx, X_t)
recommendations
```

<!-- #region id="DwlSw7xPIyYh" -->
We can't interpret the results as is since movies are represented by their index. We'll have to loop over the list of recommendations and get the movie title for each movie index. 
<!-- #endregion -->

```python id="zk__BZVMIyYh" colab={"base_uri": "https://localhost:8080/"} outputId="39577b39-27f0-41ca-8a8b-3428e7dd26ad"
for r in recommendations:
    recommended_title = get_movie_title(r[0])
    print(recommended_title)
```

<!-- #region id="AS62m5RrIyYj" -->
User 95's recommendations consist of action, crime, and thrillers. None of their recommendations are comedies. 
<!-- #endregion -->
