from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import ast
import numpy as np
import pandas as pd

# ... (your existing code for loading and processing movie data)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load your movie data and perform necessary processing
# ...

# ... (your existing code for processing movie data and generating recommendations)
# Load your movie data and perform necessary processing
# ...
credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
movies_df = movies_df.merge(credits_df, on='title')
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew' ]]
movies_df.dropna(inplace=True)
movies_df.iloc[0].genres

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)


def convert2(obj):
    L=[]
    counter= 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L

movies_df['cast'] = movies_df['cast'].apply(convert2)

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L

movies_df['crew']=movies_df['crew'].apply(fetch_director)
# movies_df['overview']= movies_df['overview'].apply(lambda x:x.split())
# movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
# movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
# movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else '' for i in x] if x else [])
# movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# movies_df['tags']=movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']

movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") if isinstance(i, str) else '' for i in x] if x else [])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast']

# # Check if 'genres' is now included in the DataFrame
# if 'genres' not in movies_df.columns:
#     print("'genres' column not found in DataFrame.")
# else:
#     print("Columns in DataFrame:", movies_df.columns)



new_df = movies_df[['movie_id', 'title', 'genres', 'tags']]
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
new_df.loc[:, 'tags'] = new_df['tags'].str.lower()




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=500, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

# Function to recommend movies by title
def recommend_by_title(movie):
    # Convert input movie title to lowercase and strip whitespaces
    movie = movie.lower().strip()

    # Check if DataFrame is empty
    if new_df.empty:
        return []

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    new_df_copy = new_df.copy()

    # Add 'title_lower' column to the DataFrame using .loc on the copy
    new_df_copy.loc[:, 'title_lower'] = new_df_copy['title'].apply(lambda x: x.lower().strip())

    # Check if the movie title is in the DataFrame (partial match)
    matching_indices = new_df_copy[new_df_copy['title_lower'].str.contains(movie)].index
    if not matching_indices.empty:
        # Use the first matching index for simplicity, you might want to refine this logic
        movie_index = matching_indices[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [new_df_copy.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    else:
        return []


# Function to recommend movies by genre
def recommend_by_genre(genre):
    # Convert input genre to lowercase and strip whitespaces
    genre = genre.lower().strip()

    # print("Input Genre:", genre)
    # print("Genres in DataFrame:", new_df['genres'])

    # Check if DataFrame is empty
    if new_df.empty:
        return []

    # Convert the DataFrame genres to lowercase lists
    lowercased_genres = new_df['genres'].apply(lambda x: [g.lower() for g in x])

    # Filter movies based on the input genre (case-insensitive match)
    genre_indices = lowercased_genres[lowercased_genres.apply(lambda x: genre in x)].index
    if not genre_indices.empty:
        # Use the first matching index for simplicity, you might want to refine this logic
        genre_index = genre_indices[0]
        distances = similarity[genre_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
        return recommended_movies
    else:
        return []



# Route to handle the first page with the option to choose recommendations by title or genre
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to handle form submission for title recommendations
@app.post("/recommend_by_title")
async def recommend_movie_by_title(request: Request, movie_title: str = Form(...)):
    recommendations = recommend_by_title(movie_title)

    if recommendations:
        return templates.TemplateResponse("recommendations.html", {"request": request, "movie_title": movie_title, "recommendations": recommendations})
    else:
        return templates.TemplateResponse("no_recommendations.html", {"request": request, "movie_title": movie_title})



    
# Route to handle form submission for genre recommendations
@app.post("/recommend_by_genre")
async def recommend_movie_by_genre(request: Request, genre: str = Form(...)):
    # Print the columns in the DataFrame
    # print("Columns in DataFrame:", new_df.columns)

    # Check if 'genres' is present in the DataFrame
    # if 'genres' not in new_df.columns:
    #     print("'genres' column not found in DataFrame.")
    #     return templates.TemplateResponse("no_recommendations.html", {"request": request, "genre": genre})

    recommendations = recommend_by_genre(genre)

    if recommendations:
        return templates.TemplateResponse("recommendations.html", {"request": request, "genre": genre, "recommendations": recommendations})
    else:
        return templates.TemplateResponse("no_recommendations.html", {"request": request, "genre": genre})



# Route for the next page
@app.get("/next_page", response_class=HTMLResponse)
def next_page(request: Request):
    return templates.TemplateResponse("submitted.html", {"request": request})