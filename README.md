# Music Recommendation System Documentation

## Overview

This Python script implements a **music recommendation system** based on **cosine similarity** between albums. It uses a dataset of albums and evaluates recommendations using **Mean Average Precision (MAP)** and **Mean Reciprocal Rank (MRR)**.

---

## Requirements

* **Python 3.x**
* Libraries:
  * `<span>pandas</span>`
  * `<span>numpy</span>`
  * `<span>scikit-learn</span>`

Install dependencies:

```
pip install pandas numpy scikit-learn
```

---

## Dataset

The dataset should be a CSV file named `<span>rym_top_5000_all_time.csv</span>` containing information about albums, including:

* **Album Name**
* **Artist Name**
* **Genres**
* **Other metadata**

---

## Code Explanation

### 1. Import Libraries

```
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
```

* **pandas**: For data manipulation.
* **numpy**: For numerical operations.
* **scikit-learn**: For similarity computation and evaluation metrics.

---

### 2. Load and Preprocess Data

```
music = pd.read_csv('rym_top_5000_all_time.csv')
```

Loads the dataset into a DataFrame.

```
music = music.drop(columns=['Descriptors', 'Ranking', 'Average Rating', 'Number of Ratings', 'Number of Reviews'])
```

Removes unnecessary columns.

```
music = pd.get_dummies(music, columns=['Genres'])
```

Encodes genres using **one-hot encoding**, transforming genres into binary columns.

---

### 3. Calculate Similarity Matrix

```
similarity = cosine_similarity(music.drop(columns=['Artist Name', 'Release Date', 'Album']))
similarity_df = pd.DataFrame(similarity, index=music['Album'], columns=music['Album'])
```

* Computes **cosine similarity** between albums based on their genre vectors.
* Stores the similarity values in a DataFrame for easy lookup.

---

### 4. Recommendation Function

```
def get_recommendation(title, similarity_df, top_n=10):
    scores = similarity_df[title].sort_values(ascending=False, kind='mergesort')
    return scores.iloc[1:top_n+1].index.tolist()
```

* **Inputs:**
  * `<span>title</span>`: Album name.
  * `<span>similarity_df</span>`: Similarity matrix.
  * `<span>top_n</span>`: Number of recommendations.
* **Outputs:**
  * A list of the most similar albums.
* **Usage:**

```
print(get_recommendation('OK Computer', similarity_df))
```

---

### 5. Evaluation Metrics

#### Reciprocal Rank

```
def reciprocal_rank(y_true, y_scores):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    for i in sorted_indices:
        if y_true[i] == 1:
            return 1 / (i + 1)
    return 0
```

Calculates the **reciprocal rank**, which evaluates how soon the first relevant recommendation appears.

#### Evaluation Function

```
def evaluation(similarity_df, top_n=10):
    average_precisions = []
    reciprocal_ranks = []

    for album in similarity_df.index:
        recommendations = get_recommendation(album, similarity_df, top_n)
        y_true = [1] + [0] * (top_n - 1)
        y_scores = [similarity_df.loc[album, rec] for rec in recommendations]
  
        average_precisions.append(average_precision_score(y_true, y_scores))
        reciprocal_ranks.append(reciprocal_rank(y_true, y_scores))

    mean_average_precision = sum(average_precisions) / len(average_precisions)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return mean_average_precision, mean_reciprocal_rank
```

* Evaluates the recommendation system using:
  * **MAP (Mean Average Precision):** Measures the quality of ranking.
  * **MRR (Mean Reciprocal Rank):** Focuses on the rank of the first relevant recommendation.
* **Usage:**

```
map_score, mrr_score = evaluation(similarity_df)
print(f'Mean Average Precision (MAP): {map_score}')
print(f'Mean Reciprocal Rank (MRR): {mrr_score}')
```

---

## Output Example

```
['Kid A', 'The Bends', 'In Rainbows', 'Hail to the Thief', 'Amnesiac']
Mean Average Precision (MAP): 0.72
Mean Reciprocal Rank (MRR): 0.85
```

---

## Limitations

1. **Input Errors:** Invalid album names cause errors. Pre-validation or fuzzy matching can address this.
3. **Evaluation Simplification:** Assumes the first recommendation should match the input album, which may not reflect real-world scenarios.

## Future Improvement

- Connecting the algorythm to the database to have proper data and value based on a user.
- Usage of other field to classify and retrieve data for the commandation system ( actually using only the genre and the field "descriptors" of the dataset
