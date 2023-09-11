import pandas as pd
import numpy as np

def get_recommendations(df, movie, similarity_measure):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[movie]

    sim_scores = list(enumerate(similarity_measure[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

def compute_precision_and_recall(df, recommendations):
    precision = 1/len(recommendations)
    recall = 1/len(df[df['vote_average'] >= 7.5])
    prec_num = 0
    rec_num = 0
    for rec in recommendations:
        if all(df[df['title'] == rec]['vote_average'] >= 7.5):
            prec_num += 1
            rec_num += 1
            
    precision *= prec_num
    recall *= rec_num
    return precision, recall

def get_random_recommendation(df, similarity_measure):
    rand_idx = np.random.randint(0, len(df))
    rand_movie = df.iloc[rand_idx]['title']
    try:
        return get_recommendations(df, rand_movie, similarity_measure), rand_movie
    except Exception:
        return get_random_recommendation(df, similarity_measure)

def main():
    df = pd.read_csv('truncated_merged_df.csv')
    sim = pd.read_csv('w2vec_sim.csv').to_numpy()

    random_recommendations, rand_movie = get_random_recommendation(df, sim)
    print(f'The recommendations for the CBOW model for the movie: {rand_movie} are:\n{random_recommendations}')

if __name__=="__main__":
    main()