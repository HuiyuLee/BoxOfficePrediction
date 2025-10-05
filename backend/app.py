from fastapi import FastAPI
import pandas as pd
from ml.BoxOfficePrediction import build_model_and_embeddings, recommend

movies_data, movie_embeddings = build_model_and_embeddings()
app = FastAPI()

# 讀取電影資料
movies_data = pd.read_csv("data/IMDB Top 250 Movies.csv")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}

@app.get("/movies/{modie_id}")
def read_movie(modie_id: int, top_k: int = 5):
    # Get movies index
    if modie_id < 0 or modie_id >= len(movies_data):
        return {"error": "Invalid movie ID"}
    movie_name = movies_data.iloc[modie_id]['name']
    similar_movie_names = recommend(modie_id, movies_data, movie_embeddings, top_k=top_k)
    return {
        "movie_id": modie_id,
        "movie_name": movie_name,
        "similar_movies": similar_movie_names
    }