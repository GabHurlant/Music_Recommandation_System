import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score

# Lecture du fichier CSV
music = pd.read_csv('rym_top_5000_all_time.csv')

# Suppression des colonnes inutiles
music = music.drop(columns=['Descriptors', 'Ranking', 'Average Rating', 'Number of Ratings', 'Number of Reviews'])

# Encodage des genres
music = pd.get_dummies(music, columns=['Genres'])

# Calcul de la similarité cosinus entre les albums
similarity = cosine_similarity(music.drop(columns=['Artist Name', 'Release Date', 'Album']))

# Création d'un DataFrame pour la matrice de similarité
similarity_df = pd.DataFrame(similarity, index=music['Album'], columns=music['Album'])

# Fonction de recommandation
def get_recommendation(title, similarity_df, top_n=10):
    # Obtenir les scores de similarité pour l'album donné
    scores = similarity_df[title].sort_values(ascending=False, kind='mergesort')
    # Obtenir les noms des albums les plus similaires
    return scores.iloc[1:top_n+1].index.tolist()

# Utilisation de la fonction de recommandation
print(get_recommendation('OK Computer', similarity_df))

# Fonction pour calculer le Reciprocal Rank
def reciprocal_rank(y_true, y_scores):
    """Calculate the Reciprocal Rank."""
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    for i in sorted_indices:
        if y_true[i] == 1:  # Trouver le premier élément pertinent
            return 1 / (i + 1)
    return 0

# Fonction d'évaluation
def evaluation(similarity_df, top_n=10):
    """Evaluate Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR)."""
    average_precisions = []
    reciprocal_ranks = []

    for album in similarity_df.index:
        # Obtenir les recommandations
        recommendations = get_recommendation(album, similarity_df, top_n)

        # Supposer que la première recommandation doit correspondre à l'album
        y_true = [1] + [0] * (top_n - 1)
        y_scores = [similarity_df.loc[album, rec] for rec in recommendations]

        # Calculer les métriques
        average_precisions.append(average_precision_score(y_true, y_scores))
        reciprocal_ranks.append(reciprocal_rank(y_true, y_scores))

    # Calculer les scores moyens
    mean_average_precision = sum(average_precisions) / len(average_precisions)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return mean_average_precision, mean_reciprocal_rank

# Évaluation du modèle
map_score, mrr_score = evaluation(similarity_df)
print(f'Mean Average Precision (MAP): {map_score}')
print(f'Mean Reciprocal Rank (MRR): {mrr_score}')