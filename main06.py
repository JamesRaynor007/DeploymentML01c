from fastapi import FastAPI, HTTPException, Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Cargar tu dataset de películas
file_path = os.path.join(os.path.dirname(__file__), 'data_reduction.csv')  # Cambia esto a la ruta de tu dataset
data = pd.read_csv(file_path)

# Inicializar el vectorizador y calcular la matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['TokensLista'])

app = FastAPI()

def get_base_url(request: Request):
    return f"{request.url.scheme}://{request.url.hostname}" + (f":{request.url.port}" if request.url.port else "")

@app.get("/")
def read_root(request: Request):
    base_url = get_base_url(request)
    return {
        "message": (
            "¡Bienvenido a la API de recomendaciones de películas!",
            "Esta API utiliza un modelo de similitud basado en TF-IDF para recomendar ",
            "películas similares a la que tú elijas.",
            "Para obtener recomendaciones, utiliza el endpoint:",
            "'/recommendations/?title={tu_titulo}'",
            "Por ejemplo, para obtener recomendaciones para 'Inception', ",
            "puedes usar la siguiente URL:",
            f"{base_url}/recommendations/?title=Inception"
        )
    }

@app.get("/recommendations/")
def get_recommendations(title: str):
    titulo_ingresado = title.lower()  # Convertir a minúsculas

    # Convertir todos los títulos del dataset a minúsculas para la comparación
    data['lower_title'] = data['title'].str.lower()

    # Verificar si la película ingresada existe en el dataset
    if titulo_ingresado not in data['lower_title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Obtener el índice de la película ingresada
    idx = data[data['lower_title'] == titulo_ingresado].index[0]

    # Calcular la similitud
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Obtener las recomendaciones
    recommendations_indices = cosine_similarities.argsort()[-6:-1][::-1]  # 5 recomendaciones

    # Combina con el voto promedio
    recommendations = data.iloc[recommendations_indices]
    recommendations['similarity'] = cosine_similarities[recommendations_indices]
    recommendations = recommendations.sort_values(by='vote_average', ascending=False)

    # Retornar solo los campos deseados
    return recommendations[['title', 'vote_average', 'similarity']].to_dict(orient='records')

# Para ejecutar la aplicación, utiliza el siguiente comando en la terminal:
# uvicorn nombre_del_archivo:app --reload
