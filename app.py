import ast
import os
import json
import requests
from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import networkx as nx

# Download NLTK data if needed
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# File paths
RECIPES_FILE = os.path.join('data', 'RAW_recipes.csv')
INGREDIENT_ORIGIN_FILE = os.path.join('data', 'ingredient_origin.csv')
df = None         # DataFrame for recipes
df_origin = None  # DataFrame for ingredient origins

def load_and_preprocess_data():
    """Load and preprocess recipes from RAW_recipes.csv."""
    global df
    df = pd.read_csv(RECIPES_FILE)
    list_columns = ['tags', 'steps', 'ingredients']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    if 'ingredients' in df.columns:
        df['ingredients'] = df['ingredients'].apply(lambda lst: [i.lower().strip() for i in lst])
    return df

def load_ingredient_origins():
    """Load ingredient origin mapping from ingredient_origin.csv."""
    global df_origin
    df_origin = pd.read_csv(INGREDIENT_ORIGIN_FILE)
    # Ensure ingredients are in lowercase and stripped
    df_origin['ingredient'] = df_origin['ingredient'].str.lower().str.strip()
    return df_origin

def compute_match(user_ingredients, recipe_ingredients):
    """Compute the match percentage between user and recipe ingredients."""
    user_set = set([i.lower().strip() for i in user_ingredients])
    recipe_set = set(recipe_ingredients)
    if not recipe_set:
        return 0, set(), set()
    common = user_set.intersection(recipe_set)
    missing = recipe_set - user_set
    match_percent = (len(common) / len(recipe_set)) * 100
    return match_percent, common, missing

def suggest_substitutions(missing_ingredients):
    """Suggest substitutions for missing ingredients."""
    substitutions = {
        'basil': 'oregano',
        'butter': 'olive oil',
        'honey': 'maple syrup',
        'salt': 'soy sauce',
        'pepper': 'chili flakes'
    }
    suggestions = {}
    for ing in missing_ingredients:
        if ing in substitutions:
            suggestions[ing] = substitutions[ing]
    return suggestions

def infer_food_identity(user_ingredients):
    """
    Infer food identity by mapping key ingredients to probable cuisines.
    Returns a sorted list of cuisines.
    """
    ingredient_to_cuisine = {
        'basil': ['Italian', 'Thai'],
        'oregano': ['Italian', 'Greek'],
        'soy sauce': ['Chinese', 'Japanese', 'Thai'],
        'ginger': ['Chinese', 'Thai', 'Indian'],
        'cumin': ['Mexican', 'Indian', 'Middle Eastern'],
        'garlic': ['Italian', 'Chinese', 'Indian'],
        'olive oil': ['Mediterranean', 'Italian'],
        'chili': ['Mexican', 'Thai'],
        'lemongrass': ['Thai', 'Vietnamese']
    }
    scores = {}
    for ing in user_ingredients:
        if ing in ingredient_to_cuisine:
            for cuisine in ingredient_to_cuisine[ing]:
                scores[cuisine] = scores.get(cuisine, 0) + 1
    if scores:
        sorted_cuisines = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cuisine for cuisine, score in sorted_cuisines]
    else:
        return ["Unknown"]

def cluster_recipes():
    """Cluster recipes using TF-IDF and KMeans to reveal recipe groupings."""
    corpus = df['ingredients'].apply(lambda lst: ' '.join(lst)).tolist()
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(corpus)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

def build_cooccurrence_network(top_n=15):
    """Build an ingredient co-occurrence network for the top N ingredients."""
    all_ing = [ing for sublist in df['ingredients'] for ing in sublist]
    top_ing = pd.Series(all_ing).value_counts().head(top_n).index.tolist()
    from collections import defaultdict
    cooccurrence = defaultdict(int)
    for ing_list in df['ingredients']:
        filtered = [ing for ing in ing_list if ing in top_ing]
        for i in range(len(filtered)):
            for j in range(i+1, len(filtered)):
                pair = tuple(sorted([filtered[i], filtered[j]]))
                cooccurrence[pair] += 1
    G = nx.Graph()
    for ing in top_ing:
        G.add_node(ing)
    for (ing1, ing2), weight in cooccurrence.items():
        if weight > 0:
            G.add_edge(ing1, ing2, weight=weight)
    return G

def generate_flavor_profile(user_ingredients):
    """
    Generate flavor profile data from user ingredients.
    Returns two lists: flavors (axes) and corresponding scores.
    """
    flavors = ['Sweet', 'Sour', 'Salty', 'Bitter', 'Umami', 'Spicy', 'Aromatic', 'Fresh']
    scores = {flavor: 0 for flavor in flavors}
    flavor_mapping = {
        'sugar': 'Sweet',
        'honey': 'Sweet',
        'lemon': 'Sour',
        'vinegar': 'Sour',
        'salt': 'Salty',
        'soy sauce': 'Umami',
        'chili': 'Spicy',
        'pepper': 'Spicy',
        'basil': 'Aromatic',
        'garlic': 'Aromatic',
        'mint': 'Fresh'
    }
    for ing in user_ingredients:
        for key, flavor in flavor_mapping.items():
            if key in ing:
                scores[flavor] += 1
    return flavors, list(scores.values())

def generate_cultural_origin_map(user_ingredients):
    """
    Generate map data for an interactive world map that shows the origins
    of the user-input ingredients based on df_origin.
    """
    if df_origin is None:
        return None
    df_filtered = df_origin[df_origin['ingredient'].isin(user_ingredients)]
    region_counts = df_filtered['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    df_regions = df_filtered.groupby('region').agg({'lat': 'mean', 'lon': 'mean'}).reset_index()
    map_data = pd.merge(region_counts, df_regions, on='region')
    return map_data

@app.route('/')
def index():
    """Render the landing page for ingredient input."""
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    """Process the user input, match recipes, infer food identity, and generate visualizations."""
    user_input = request.form.get('ingredients', '')
    user_ingredients = [i.strip().lower() for i in user_input.split(',') if i.strip()]
    
    recipes_list = []
    for idx, row in df.iterrows():
        match_percent, have, missing = compute_match(user_ingredients, row['ingredients'])
        if match_percent >= 10:  # Lower threshold to include even low-match recipes
            subs = suggest_substitutions(missing)
            recipes_list.append({
                'name': row['name'],
                'match_percent': round(match_percent, 2),
                'ingredients': row['ingredients'],
                'have': list(have),
                'missing': list(missing),
                'substitutions': subs,
                'steps': row['steps'],
                'description': row['description']
            })
    # Sort recipes by match percentage descending
    filtered_recipes = sorted(recipes_list, key=lambda x: x['match_percent'], reverse=True)
    
    # Infer food identity
    food_identity = infer_food_identity(user_ingredients)
    
    # Cultural Origin Map Data
    map_data = generate_cultural_origin_map(user_ingredients)
    if map_data is not None:
        fig_map = px.scatter_geo(map_data,
                                 lat='lat',
                                 lon='lon',
                                 size='count',
                                 hover_name='region',
                                 projection="natural earth",
                                 title="Cultural DNA of Your Ingredients")
        graphJSON_map = json.dumps(fig_map, cls=PlotlyJSONEncoder)
    else:
        graphJSON_map = None

    # Recipe Match Bar Chart
    if filtered_recipes:
        chart_df = pd.DataFrame({
            'Recipe': [r['name'] for r in filtered_recipes],
            'Match Percentage': [r['match_percent'] for r in filtered_recipes]
        })
        fig_match = px.bar(chart_df,
                           x='Recipe',
                           y='Match Percentage',
                           title='Recipe Matches',
                           template='plotly_dark')
        fig_match.update_layout(title={'x': 0.5, 'xanchor': 'center'},
                                xaxis_tickangle=-45)
        graphJSON_match = json.dumps(fig_match, cls=PlotlyJSONEncoder)
    else:
        graphJSON_match = None

    # Top Ingredient Frequency Bar Chart (from whole dataset)
    all_ingredients = [ing for sublist in df['ingredients'] for ing in sublist]
    freq = pd.Series(all_ingredients).value_counts().head(10)
    fig_ing = px.bar(x=freq.index, y=freq.values,
                     labels={'x': 'Ingredient', 'y': 'Frequency'},
                     title="Top 10 Ingredients in Our Recipes",
                     template='plotly_white')
    graphJSON_ing = json.dumps(fig_ing, cls=PlotlyJSONEncoder)

    # Recipe Clusters Pie Chart
    if 'cluster' in df.columns:
        cluster_counts = df['cluster'].value_counts()
        fig_cluster = px.pie(names=cluster_counts.index.astype(str),
                             values=cluster_counts.values,
                             title="Recipe Clusters & Cultural Flavors",
                             template='plotly_white')
        graphJSON_cluster = json.dumps(fig_cluster, cls=PlotlyJSONEncoder)
    else:
        graphJSON_cluster = None

    # Ingredient Co-occurrence Network Graph
    G = build_cooccurrence_network(top_n=15)
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = {
        'x': edge_x,
        'y': edge_y,
        'line': {'width': 1, 'color': '#888'},
        'hoverinfo': 'none',
        'mode': 'lines',
        'type': 'scatter'
    }
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = {
        'x': node_x,
        'y': node_y,
        'mode': 'markers+text',
        'text': node_text,
        'textposition': 'top center',
        'marker': {'size': 15, 'color': '#FFA07A', 'line': {'width': 2, 'color': '#555'}},
        'type': 'scatter'
    }
    fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Ingredient Co-occurrence Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))
    graphJSON_network = json.dumps(fig_network, cls=PlotlyJSONEncoder)

    # Flavor Profile Radar Chart
    flavors, flavor_scores = generate_flavor_profile(user_ingredients)
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=flavor_scores,
        theta=flavors,
        fill='toself',
        name='Flavor Profile'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(flavor_scores) + 1])),
        title="Your Flavor Fingerprint",
        template="plotly_dark"
    )
    graphJSON_radar = json.dumps(fig_radar, cls=PlotlyJSONEncoder)

    return render_template('results.html',
                           recipes=filtered_recipes,
                           user_ingredients=user_ingredients,
                           food_identity=food_identity,
                           graphJSON_map=graphJSON_map,
                           graphJSON_match=graphJSON_match,
                           graphJSON_ing=graphJSON_ing,
                           graphJSON_cluster=graphJSON_cluster,
                           graphJSON_network=graphJSON_network,
                           graphJSON_radar=graphJSON_radar)

if __name__ == '__main__':
    load_and_preprocess_data()
    load_ingredient_origins()
    cluster_recipes()
    app.run(debug=True)
