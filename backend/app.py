from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
from ml.BoxOfficePrediction import build_model_and_embeddings, recommend
import os

app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS) to allow frontend applications from different origins to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
movies_data, movie_embeddings = build_model_and_embeddings()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Endpoint to retrieve the list of all movies with their IDs and names
@app.get("/movies_list")
def get_movies_list():
    return [{"movie_id": idx, "movie_name": row['name']} for idx, row in movies_data.iterrows()]

# Endpoint to get a specific movie's details and its top-K similar movie recommendations
@app.get("/movies/{movie_id}")
def read_movie(movie_id: int, top_k: int = 5):
    # Get movies index
    if movie_id < 0 or movie_id >= len(movies_data):
        return {"error": "Invalid movie ID"}
    movie_name = movies_data.iloc[movie_id]['name']
    similar_movie_names = recommend(movie_id, movies_data, movie_embeddings, top_k=top_k)
    return {
        "movie_id": movie_id,
        "movie_name": movie_name,
        "similar_movies": similar_movie_names
    }