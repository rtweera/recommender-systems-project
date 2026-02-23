# Books Recommender

This is a comprehensive recommender system project built using Python, leveraging libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib. The project focuses on building a book recommendation system using the Book-Crossing dataset.

The project includes the following key components:

1. **Data Loading and Preprocessing**: The dataset is loaded and preprocessed to handle missing values, filter out users and books with insufficient ratings, and split the data into training and testing sets.
2. **Baseline Models**: Simple baseline recommendation models are implemented, including a popular items model and a random recommendation model.
3. **Evaluation Metrics**: The performance of the recommendation models is evaluated using metrics such as Precision@10, Recall@10, NDCG@10, Hit Rate@10, Personalization, and Coverage.
4. **Demonstration**: The system is demonstrated on a few warm-start and cold-start users to showcase the recommendations.
5. **Visualization**: The results are visualized using Matplotlib and Seaborn to provide insights into the performance of the models.

## Getting Started

To run this project, you will need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install scikit-learn matplotlib seaborn networkx[default] ipywidgets tqdm numpy pandas
```

Make sure to download the Book-Crossing dataset and place it in the appropriate directory (`data/`) as specified in the code.

Data can be found here: [Book-Crossing Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data?select=Users.csv)

## Project Structure

- recommender.ipynb: The main Jupyter notebook containing the implementation of the recommender system.
- Data_Link.md: A markdown file containing the link to the dataset used in this project.
- README.md: This file, providing an overview of the project and instructions for running it.
- 215565L_recommender.mp4: A video explaining the project and its results.

## Algorithms Explained

- **Content-based**: Recommends items similar to those the user has liked in the past, based on item features (title, author, published year). Captures user's unique taste. Also useful when user has few ratings (cold-start). (Coz difficult to match the user with similar users if they have few ratings - can be highly biased if we did).
- **Collaborative Filtering**: Recommends items based on the preferences of similar users (user-based). Can capture complex patterns but struggles with cold-start users.
- **PageRank**: Finds the most popular books so these can be recommended when we have low information about the user (cold-start).
- **Personalization**: Boost scores based on age. Young people will love new books, while older people will prefer classics. (assumption based on common stereotypes, but it can be a useful heuristic in some cases).

> So generally we use a mix of content-based and collaborative on 30% to 70% basis, so capture best of both worlds. For cold-start users, we rely more on content-based and popularity-based recommendations. For warm users, we can leverage collaborative filtering more effectively. In any case we do personalization to boost scores based on age.

## Evaluation Metrics Explained

### Normal metrics

- **Precision@10**: Of the top 10 recommendations, how many did the user actually like?
- **Recall@10**: Of all the books the user liked, how many did you find?
- **NDCG@10**: Ideal ranking metric where the position of the correct recommendations
    matters (higher is better).
- **Hit Rate@10**: Did you get at least 1 correct recommendation in the top 10?

### Advanced metrics

- **Personalization**: How different are the recommendation lists across users? (higher is more personalized)
- **Coverage**: What percentage of all books is recommended overall? (higher is better coverage)

### Expected Results

Our model should outperform all the baselines in all metrics except in personalization and coverage. In those cases, the random baseline will likely have the highest scores because it generates recommendations without any bias towards popular items or user preferences, leading to high diversity in recommendations across users and a wide variety of items being recommended (but been garbage recommendations).

> **NOTE:** The graphs in the notebook are log-scaled to better visualize the differences of small values.

### Why Random Baseline Has High Personalization and coverage?

The random baseline can have artificially high personalization and coverage because it generates recommendations without any bias towards popular items or user preferences. This means that it can recommend a wide variety of items across different users, leading to high coverage. Additionally, since the recommendations are random, the overlap between different users' recommendation lists is likely to be low, resulting in high personalization.

> **NOTE:** However, it's important to note that while the random baseline may score well on these metrics, it is not a useful recommendation system, as it does not provide meaningful recommendations to users.
