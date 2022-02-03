# Tasks

## Next Basket Recommendation

Next Basket Recommendation (NBR) is a sequential recommendation task where the goal is to recommend a set of items based on a user’s purchase history. NBR is of great interest to the e-commerce and retail industry, where we want to recommend a set of items to fill a user’s shopping basket. A basket ***b*** is a set of items, i.e., ***b*** = $\{𝑖_1,𝑖_2,\dots,𝑖_𝑗,\dots,𝑖_{|𝒃|}\}$, where $𝑖_𝑗 \in I$, and where $I$ denotes the universe of all items. For a given user, we have access to a sequence of 𝑛 historical baskets (in increasing chronological order, such that more recent items are at the tail) denoted as $H = [𝒃_1, 𝒃_2, \dots, 𝒃_𝒊 , \dots, 𝒃_𝒏]$, where $𝒃_i \subset I$. The goal of NBR is then to a find a model which takes the historical baskets $H$ as input and predicts the next basket $𝒃_{𝒏+1}$ as recommendation.
