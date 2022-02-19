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

<!-- #region id="yH-WcJpQKox7" -->
# Diversity Aware Book Recommender
> A tutorial on building an amazon-like book recommender and keeping diversity as an important factor

- toc: true
- badges: true
- comments: true
- categories: [diversity, book]
- image: 
<!-- #endregion -->

```python id="WafZxDxkKK27"
import math
import heapq

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
```

```python id="hOkyd4lZLGre"
books_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/books.csv", encoding="ISO-8859-1")
books_tags_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/books_tags.csv", encoding="ISO-8859-1")
users_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/users.csv", encoding="ISO-8859-1")
ratings_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/ratings.csv", encoding="ISO-8859-1")
tags_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/tags.csv", encoding="ISO-8859-1")
test_df = pd.read_csv("https://raw.githubusercontent.com/sparsh-ai/reco-data/master/goodreads/test.csv", encoding="ISO-8859-1")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 428} id="q0ZCy-0jMOgW" outputId="99b07cf2-ecf1-4dda-eb13-0e8f4f1ac194"
books_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6SaAmNSEYyg9" outputId="23430e4c-da64-45f6-d954-1f815a50431e"
books_df.columns
```

```python colab={"base_uri": "https://localhost:8080/"} id="aeFq3hfnY38U" outputId="2e202f0e-5967-4f37-80bc-5ca03c9b1f94"
books_df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="bShG86UoMog9" outputId="fa6b5047-eae2-42e2-c626-efbb237f1a62"
books_tags_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="g0dpPnj9MoN1" outputId="7d831402-e55e-4263-8778-018d06ffb4be"
users_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="e00fRG13MoCy" outputId="66a42f46-ea16-4c39-bea1-ffdc40d4525a"
ratings_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="pCxaDwXOMnRY" outputId="a3125d83-05ff-4348-d951-2bc33aeda18d"
tags_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8imx2ekDMvJa" outputId="05165568-a222-4c65-9d8c-811bed89a781"
test_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="S0NoZqP3L-bA" outputId="0c988ab2-e625-4ec1-fb9e-ff219197d2d2"
b_r_u_df = pd.merge(books_df[['book_id', 'title']], ratings_df, on='book_id', how='inner')
b_r_u_df = pd.merge(b_r_u_df, users_df, on='user_id', how='inner')
items_df = b_r_u_df[['book_id', 'title']].drop_duplicates(subset='book_id')
items_df.head()
```

<!-- #region id="doeC66SkRRRD" -->
## Util functions
<!-- #endregion -->

```python id="PhHNANjcNCS4"
# Score for each book
def weighted_average_rating(counts_df, m, rating_mean_df, rating_all_mean):
    return (counts_df / (counts_df + m)) * rating_mean_df + (m / (counts_df + m)) * rating_all_mean


# To get the top k recommendations, the k books with the best score
def get_top_k_recommendations(df, counts_df, k, columns, m):
    return df[counts_df >= m].nlargest(k, columns)


# Get the top k recommendations: the k books with the best weighted average rating
def calculate_WAR_and_top_k(df, m, rating_all_mean, k, columns):
    df['weighted_average_rating'] = weighted_average_rating(df['counts'], m, df['rating_mean'], rating_all_mean)
    return get_top_k_recommendations(df, df['counts'], k, columns, m)


# Gets a number and creates the range of ages as required in the task
def get_age_range(age):
    if age % 10 == 0:
        age -= 1
    lower_bound = age - ((age % 10) - 1)
    upper_bound = age + (10 - (age % 10))
    return lower_bound, upper_bound


# Creates distribution of votes and ratings by book id and returns them
def merge_tables(df):
    # Dataframe that contains distribution of votes by book ID
    nb_voters_data = df['book_id'].value_counts()
    nb_voters_df = pd.DataFrame(data={'book_id': nb_voters_data.index.tolist(), 'counts': nb_voters_data.values.tolist()})

    # Dataframe that contains distribution of rate averages by book ID
    rating_mean_data = df.groupby(['book_id'])['rating'].mean()
    rating_mean_df = pd.DataFrame(data={'book_id': rating_mean_data.index.tolist(), 'rating_mean': rating_mean_data.values.tolist()})
    
    return nb_voters_df, nb_voters_data, rating_mean_df, rating_mean_data


# m represents the minimum voters we need to count the rating and the score
# We'll also need the total mean to caluculate the score (WR)
def get_voters_and_means(nb_voters_data, rating_mean_df):
    m = nb_voters_data.quantile(0.90)
    rating_all_mean = rating_mean_df['rating_mean'].mean()
    return m, rating_all_mean
```

<!-- #region id="GWBsBlhyRUy4" -->
## Non-personalized recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="90UPDNr5OJde" outputId="e3242afe-caab-4c12-9d84-d91ca6f13ecc"
# Calculate the WAR and get the k books with the best results
def get_simply_recommendation(k):

    nb_voters_df, nb_voters_data, rating_mean_df, rating_mean_data = merge_tables(b_r_u_df)
    m, rating_all_mean = get_voters_and_means(nb_voters_data, rating_mean_df)
    
    df = pd.merge(items_df, nb_voters_df, on='book_id', how='inner')
    df = pd.merge(df, rating_mean_df, on='book_id', how='inner')

    return calculate_WAR_and_top_k(df, m, rating_all_mean, k, ['weighted_average_rating'])


recommendation_df = get_simply_recommendation(10)
recommendation_df[['book_id','title','weighted_average_rating']].head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="YaRGlLDCOmZz" outputId="6a668723-5265-4131-eed2-bbe7d1c314ab"
# Calculate the WAR and get the k best books by a place
def get_simply_place_recommendation(place, k):
    global b_r_u_df
    
    b_r_u_place_df = b_r_u_df[b_r_u_df['location'] == place]
    
    nb_voters_df, nb_voters_data, rating_mean_df, rating_mean_data = merge_tables(b_r_u_place_df)
    m, rating_all_mean = get_voters_and_means(nb_voters_data, rating_mean_df)
    
    df = pd.merge(items_df, nb_voters_df, on='book_id', how='inner')
    df = pd.merge(df, rating_mean_df, on='book_id', how='inner')

    return calculate_WAR_and_top_k(df, m, rating_all_mean, k, ['weighted_average_rating'])


place_recommendation_df = get_simply_place_recommendation('Ohio', 10)
place_recommendation_df[['book_id','title','weighted_average_rating']].head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="ApwrZaisOvPe" outputId="06b7c576-5aa4-44f7-e8ee-d8a626ac6a77"
# Calculate the WAR and get the k best books by a range of ages
def get_simply_age_recommendation(age, k):
    global b_r_u_df
    
    lower_bound, upper_bound = get_age_range(age)
    b_r_u_age_df = b_r_u_df[(b_r_u_df['age'] >= lower_bound) & (b_r_u_df['age'] <= upper_bound)]
    
    nb_voters_df, nb_voters_data, rating_mean_df, rating_mean_data = merge_tables(b_r_u_age_df)
    m, rating_all_mean = get_voters_and_means(nb_voters_data, rating_mean_df)
    
    df = pd.merge(items_df, nb_voters_df, on='book_id', how='inner')
    df = pd.merge(df, rating_mean_df, on='book_id', how='inner')

    return calculate_WAR_and_top_k(df, m, rating_all_mean, k, ['weighted_average_rating'])


age_recommendation_df = get_simply_age_recommendation(28, 10)
age_recommendation_df[['book_id','title','weighted_average_rating']].head(10)
```

<!-- #region id="LRMn2824RYgj" -->
## Collaborative filtering
<!-- #endregion -->

```python id="RQM0lI8NO01_"
# Get top K values from given array
def keep_top_k(array, k):
    smallest = heapq.nlargest(k, array)[-1]
    array[array < smallest] = 0
    return array


# Similirity matrix according to the chosen similarity
def build_CF_prediction_matrix(sim):
    global ratings_diff
    return 1-pairwise_distances(ratings_diff, metric=sim)


# Function that extracts user recommendations
# Gets the highest rates indexes and returns book id and the title for that user
def get_CF_final_output(pred, data_matrix, user_id, items_new_to_original, items, k):
    user_id = user_id - 1
    predicted_ratings_row = pred[user_id]
    data_matrix_row = data_matrix[user_id]
    
    predicted_ratings_unrated = predicted_ratings_row.copy()
    predicted_ratings_unrated[~np.isnan(data_matrix_row)] = 0

    idx = np.argsort(-predicted_ratings_unrated)
    sim_scores = idx[0:k]
    
    # When getting the results, we map them back to original book ids
    books_original_indexes_df = pd.DataFrame(data={'book_id': [items_new_to_original[index] for index in sim_scores]})
    return pd.merge(books_original_indexes_df, items, on='book_id', how='inner')
```

```python id="yJBv6OOwRuF9"
# Get the collaborative filtering recommendation according to the user
def get_CF_recommendation(user_id, k):
    # Import global variables
    global ratings_df
    global items_df

    # Declare of global variables
    global users_original_to_new
    global users_new_to_original
    global items_original_to_new
    global items_new_to_original
    global data_matrix
    global mean_user_rating

    # Part 1
    unique_users = ratings_df['user_id'].unique()
    unique_items = ratings_df['book_id'].unique()
    n_users = unique_users.shape[0]
    n_items = unique_items.shape[0]

    # Working on user data
    unique_users.sort()
    # Creating a dictionary that contains a mapping from original user id to a new user id
    users_original_to_new = {original_index: new_index for original_index, new_index in zip(unique_users, range(n_users))}
    # Creating a dictionary that contains a mapping from new user id to a original user id
    users_new_to_original = {value: key for key, value in users_original_to_new.items()}

    # Working on items data
    unique_items.sort()
    # Creating a dictionary that contains a mapping from original book id to a new book id
    items_original_to_new = {original_index: new_index for original_index, new_index in zip(unique_items, range(n_items))}
    # Creating a dictionary that contains a mapping from new book id to a original book id
    items_new_to_original = {value: key for key, value in items_original_to_new.items()}

    # Part 2
    data_matrix = np.empty((n_users, n_items))
    data_matrix[:] = np.nan
    for line in ratings_df.itertuples():
        user = users_original_to_new[line[1]]
        book = items_original_to_new[line[2]]
        rating = line[3]
        data_matrix[user, book] = rating

    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

    global ratings_diff
    ratings_diff = (data_matrix - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)] = 0

    user_similarity = build_CF_prediction_matrix('cosine')
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    # Part 3
    return get_CF_final_output(pred, data_matrix, user_id, items_new_to_original, items_df, k)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="0eKgqb_6R6HQ" outputId="cd194361-dd6e-46a3-9e41-def63f1094de"
recommendations_by_user = get_CF_recommendation(user_id=1, k=10)
recommendations_by_user.head(10)
```

<!-- #region id="cpy4KaJKSJk_" -->
## Content-based filtering
<!-- #endregion -->

<!-- #region id="LfHfZc3WW7Gl" -->
The features we used to work with are: language, tags, original title and authors.

We tried different features to choose the best ones. It seems logical that if we want a recommendation for Twilight or Harry Potter, or The Hunger Games, it will recommend us the other books of the Saga. That’s why the title’s book is important.

Moreover, usually, every author has its own writing style. We can see it through their different books. And if I liked a book of an author, I would like to get recommendations for its other books.

Language has also its logical impact. The way it’s been written, the language the lector speaks and language can also mean culture. French books are different from American books.

Tags give us some hints about the books: like the genre. It’s also important for the recommendation. We also tried with the publication year but it adds noises. It limits us. The year is taken as a string, so it’s either equal or different. That is not what we want.
<!-- #endregion -->

```python id="BUap3tjhR9pW"
# Preparing the dataframe for the algorithm
# Reading the tags and the relevant features and preparing the features for the algorithm
bookreads_tags_df = pd.merge(books_tags_df, tags_df, on='tag_id', how='inner')

groupped_data = bookreads_tags_df.groupby('goodreads_book_id', as_index=False)['tag_name'].transform(lambda x: ' '.join(x))
books_tags_row_df = pd.DataFrame(data={'goodreads_book_id': groupped_data.index.tolist(), 'tag_name': groupped_data['tag_name'].values.tolist()})

content_based_filtering_df = pd.merge(books_df[['book_id', 'title', 'authors', 'goodreads_book_id',  'language_code', 'original_title']], books_tags_row_df, on='goodreads_book_id', how='outer')
content_based_filtering_df['tag_name'] = content_based_filtering_df['tag_name'].fillna('')
```

```python id="ni1k-NLySaub"
# Clean the data to get lower case letter and get rid of '-'
def clean_data(x):
    x = str.lower(str(x))
    return x.replace('-', '')


# Get all of our features together with a space in between. The choice of the features is explained in the report.
def create_soup(x):
    return x['original_title'] + ' ' + x['language_code'] + ' ' + x['authors'] + ' ' + x['tag_name']


# Similarity matrix. We use cosine similarity
def build_contact_sim_metrix():
    global count_matrix
    return cosine_similarity(count_matrix, count_matrix)
```

```python id="Y6JKv69GSdbf"
# We return the top k recommendation according to the content (The features of each book)
def get_content_recommendation(book_name, k):
    global content_based_filtering_df

    features = ['original_title', 'language_code', 'authors', 'tag_name']
    for feature in features:
        content_based_filtering_df[feature] = content_based_filtering_df[feature].apply(clean_data)

    content_based_filtering_df['soup'] = content_based_filtering_df.apply(create_soup, axis=1)

    global count_matrix
    count_matrix = CountVectorizer(stop_words='english').fit_transform(content_based_filtering_df['soup'])

    cosine_sim = build_contact_sim_metrix()

    content_based_filtering_df = content_based_filtering_df.reset_index()
    indices = pd.Series(content_based_filtering_df.index, index=content_based_filtering_df['title'])

    idx = indices[book_name]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]

    book_indices = [i[0] for i in sim_scores]
    return content_based_filtering_df['title'].iloc[book_indices]
```

```python id="kPt_aef5Sg_e"
content_based_filtering_result = get_content_recommendation('Twilight (Twilight, #1)', k=10)
content_based_filtering_result
```

<!-- #region id="7HWTE8oFTcc1" -->
## Evaluations
<!-- #endregion -->

```python id="1bOaw7v7Sllp"
# Filter the data
# We want only the books with ratings >= 4 and the users that that ranked books at least 10 times
high_rate_test_df = test_df[test_df['rating'] >= 4]
user_value_counts = high_rate_test_df['user_id'].value_counts()
user_value_counts_df = pd.DataFrame(data={'user_id': user_value_counts.index.tolist(), 'appearances': user_value_counts.values.tolist()})
user_value_counts_df = user_value_counts_df[user_value_counts_df['appearances'] >= 10]
```

<!-- #region id="Hkksp1zPWbU6" -->
We get a weak precision_k for every similarity. The reason to this is that the test file is very small so it can’t give us a good precision. We don’t have enough information. We could have got better results for ARHR, but we didn’t for the same reasons. If we had more samples in our test file, the ARHR would be better. We still get better results than precision_k because we take into account the position of the books. Moreover, precision_k and ARHR, use only the top 10 recommendations that have been given. RMSE takes into account the predicted results and compare it to the actual one. Only the difference between the rankings matters. Those results show us that our rankings are good and that we succeeded to find the right rank of the recommendation (ARHR high).
<!-- #endregion -->

<!-- #region id="6OFEzfCaWnTQ" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iiyfRK5FThOF" outputId="ab637b85-aa47-488a-e0c1-636298550343"
# Evaluation P@k for each similarity
def precision_k(k):
    # Importing global variables that has been assigned before to use them here
    global items_df
    global data_matrix
    global ratings_diff
    global mean_user_rating
    global items_new_to_original
    
    pk_list = []
    for sim in ['cosine', 'euclidean', 'jaccard']:
        calculations_list = []

        user_similarity = 1-pairwise_distances(ratings_diff, metric=sim)
        user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

        for user_id in user_value_counts_df['user_id'].values:
            user_recommendations = get_CF_final_output(pred, data_matrix, user_id, items_new_to_original, items_df, k)

            counter = 0
            for book_idx in user_recommendations['book_id'].values:
                chosen_book = high_rate_test_df[(high_rate_test_df['user_id'] == user_id) & (high_rate_test_df['book_id'] == book_idx)]
                if chosen_book.shape[0] == 1:
                    counter += 1

            calculations_list.append(counter / k)

        pk_list.append(sum(calculations_list) / user_value_counts_df.shape[0])

    return pk_list


print(precision_k(10))
```

```python colab={"base_uri": "https://localhost:8080/"} id="bC4QPTjeUheO" outputId="dcb1f4e4-7a8a-423b-e954-4c318563369b"
# Evaluation ARHR
def ARHR(k):
    # Importing global variables that has been assigned before to use them here
    global items_df
    global data_matrix
    global ratings_diff
    global mean_user_rating
    global items_new_to_original
    
    arhr_list = []
    for sim in ['cosine', 'euclidean', 'jaccard']:
        calculations_list = []

        user_similarity = 1-pairwise_distances(ratings_diff, metric=sim)
        user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

        for user_id in user_value_counts_df['user_id'].values:
            user_recommendations = get_CF_final_output(pred, data_matrix, user_id, items_new_to_original, items_df, k)

            user_high_rate_df = high_rate_test_df[high_rate_test_df['user_id'] == user_id]
            user_rec_merged_df = pd.merge(user_recommendations, user_high_rate_df, on='book_id', how='inner')

            for position in user_rec_merged_df.index + 1:
                calculations_list.append(1 / position)

        arhr_list.append(sum(calculations_list) / user_value_counts_df.shape[0])

    return arhr_list


ARHR(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="VIrIaB0rUnMo" outputId="981642cb-41cb-40c0-b13c-5f57834155a4"
# Helper function to build the function RMSE. This time we wont filter the data.
def get_recommendations_RMSE(pred, data_matrix, user_id):
    user_id = user_id - 1
    predicted_ratings_row = pred[user_id]
    data_matrix_row = data_matrix[user_id]

    predicted_ratings_unrated = predicted_ratings_row.copy()
    predicted_ratings_unrated[~np.isnan(data_matrix_row)] = 0

    book_ids = np.argsort(-predicted_ratings_unrated)
    books_rating = np.sort(predicted_ratings_unrated)[::-1]

    return {idx: rating for idx, rating in zip(book_ids, books_rating)}


# Evaluation RMSE.
def RMSE():
    # Importing global variables that has been assigned before to use them here.
    global ratings_diff
    global mean_user_rating
    
    rmse_list = []
    for sim in ['cosine', 'euclidean', 'jaccard']:
        sum_error = 0
        count_lines = 0

        user_similarity = 1-pairwise_distances(ratings_diff, metric=sim)
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

        for user_id, test_user_data in test_df.groupby('user_id'):
            user_recommendations = get_recommendations_RMSE(pred, data_matrix, user_id)

            for row in test_user_data.itertuples(index=False):
                _, test_book_id, rating = tuple(row)
                prediction = user_recommendations[test_book_id] if test_book_id in user_recommendations else 0

                if prediction == 0:
                    continue

                sum_error += (prediction - rating)**2
                count_lines += 1

        rmse_list.append(math.sqrt(sum_error/count_lines))

    return rmse_list
    
    
RMSE()
```

```python id="Qm8lnBcBVZFN"

```
