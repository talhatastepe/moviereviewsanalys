import json

# Read the JSON file
with open('letterboxd_movies.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Replace NaN values with null
for movie in data:
    if 'release_year' in movie and movie['release_year'] == 'NaN':
        movie['release_year'] = None
    if 'average_rating' in movie and movie['average_rating'] == 'NaN':
        movie['average_rating'] = None

# Write the updated data back to the file
with open('letterboxd_movies.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
