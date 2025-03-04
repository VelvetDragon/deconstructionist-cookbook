import ast
import os
import json
from markupsafe import Markup
from flask import Flask, render_template, request, url_for
import pandas as pd
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import networkx as nx
import requests  # For Wikipedia API

# Download NLTK data once (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)

# File path
RAW_RECIPES_FILE = os.path.join('data', 'RAW_recipes.csv')

# Global DataFrame and precomputed visualizations
df_raw = None
precomputed_visualizations = {}

def load_raw_recipes():
    """Load and cache RAW_recipes.csv, optimizing for performance."""
    global df_raw
    if df_raw is None:
        df_raw = pd.read_csv(RAW_RECIPES_FILE, nrows=5000)  # Limit rows for faster loading (adjust as needed)
        # Convert string representations of lists for certain columns
        for col in ['tags', 'steps', 'ingredients']:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
        if 'ingredients' in df_raw.columns:
            df_raw['ingredients'] = df_raw['ingredients'].apply(lambda lst: [i.lower().strip() for i in lst])
        # Ensure name is a string, replacing NaN with empty string
        df_raw['name'] = df_raw['name'].fillna('').astype(str)
    return df_raw

def precompute_visualizations():
    """Precompute simplified visualizations with enhanced food identity elements and correct Plotly config."""
    global precomputed_visualizations
    if not precomputed_visualizations:
        df = load_raw_recipes()
        df_stats = compute_ingredients_stats(df)
        
        # Top 10 Ingredient Frequency (Simplified)
        all_ingredients = [ing for sublist in df['ingredients'] for ing in sublist]
        freq = pd.Series(all_ingredients).value_counts().head(10)  # Reduced to top 10 for simplicity
        fig_freq = px.bar(x=freq.index, y=freq.values,
                          labels={'x': 'Ingredient', 'y': 'Usage Count'},
                          title="Top Ingredients Reflecting Your Food Identity",
                          template='plotly_white')
        fig_freq.update_layout(showlegend=False)  # Remove legend
        freq_description = "This simple bar chart shows the 10 most common ingredients, linking them to food identities. For example, basil signals Italian or Thai heritage, while cumin points to Mexican or Indian roots, revealing your culinary preferences."
        # Serialize with PlotlyJSONEncoder, config handled in frontend
        precomputed_visualizations['freq'] = {
            'json': fig_freq.to_json(),
            'description': freq_description
        }
        
        # Histogram of Number of Ingredients per Recipe (Simplified)
        fig_hist = px.histogram(df_stats, x='num_ingredients',
                                nbins=10,  # Fewer bins for simplicity
                                title="How Recipe Size Hints at Your Culinary Culture",
                                labels={'num_ingredients': 'Ingredients'},
                                template='plotly_white')
        fig_hist.update_layout(showlegend=False)
        hist_description = "This bar chart shows how many ingredients recipes typically use, suggesting your food identity. Fewer ingredients might indicate Italian simplicity, while more suggest vibrant Mexican or Indian diversity, reflecting your cooking style."
        precomputed_visualizations['hist'] = {
            'json': fig_hist.to_json(),
            'description': hist_description
        }
        
        # Scatter Plot of Instruction Length vs. Number of Ingredients (Simplified)
        fig_scatter = px.scatter(df_stats, x='num_ingredients', y='instruction_length',
                                 title="Your Cooking Style: Simplicity or Complexity?",
                                 labels={'num_ingredients': 'Ingredients', 'instruction_length': 'Steps'},
                                 template='plotly_white')
        fig_scatter.update_layout(showlegend=False)
        scatter_description = "This scatter plot shows how many ingredients and steps recipes have, revealing your food identity. Short steps with few ingredients might suggest Japanese minimalism, while many ingredients with detailed steps could point to French or Indian richness."
        precomputed_visualizations['scatter'] = {
            'json': fig_scatter.to_json(),
            'description': scatter_description
        }
        
        # Ingredient Co-occurrence Network Graph (Simplified for Food Identity)
        ingredient_to_cuisine = {
            'basil': 'Italian/Thai', 'oregano': 'Italian/Greek', 'soy sauce': 'Japanese/Chinese/Korean',
            'tortilla': 'Mexican', 'ginger': 'Chinese/Thai/Indian', 'cumin': 'Mexican/Indian/Middle Eastern',
            'cilantro': 'Mexican/Thai/Indian', 'parmesan': 'Italian', 'sriracha': 'Thai',
            'curry': 'Indian/Thai'
        }
        all_ing = [ing for sublist in df['ingredients'] for ing in sublist]
        top_ing = pd.Series(all_ing).value_counts().head(8).index.tolist()  # Reduced to top 8 for clarity
        G = nx.Graph()
        for ing in top_ing:
            cuisine = ingredient_to_cuisine.get(ing, 'General')
            G.add_node(ing, cuisine=cuisine)
        cooccurrence = {}
        for ing_list in df['ingredients']:
            filtered = [ing for ing in ing_list if ing in top_ing]
            for i in range(len(filtered)):
                for j in range(i+1, len(filtered)):
                    pair = tuple(sorted([filtered[i], filtered[j]]))
                    cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        for (ing1, ing2), weight in cooccurrence.items():
            if weight > 0:
                G.add_edge(ing1, ing2, weight=weight)
        
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = {
            'x': edge_x, 'y': edge_y, 'line': {'width': 1, 'color': '#888'},
            'hoverinfo': 'none', 'mode': 'lines', 'type': 'scatter'
        }
        node_x, node_y, node_text, node_colors = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            cuisine = G.nodes[node]['cuisine']
            color = '#FFA07A' if 'Italian' in cuisine or 'Mexican' in cuisine else '#98FB98' if 'Thai' in cuisine else '#DDA0DD' if 'Indian' in cuisine else '#87CEEB'
            node_colors.append(color)
            node_text.append(f"{node} ({cuisine})")
        node_trace = {
            'x': node_x, 'y': node_y, 'mode': 'markers+text', 'text': node_text, 'textposition': 'top center',
            'marker': {'size': 8, 'color': node_colors, 'line': {'width': 1, 'color': '#555'}}, 'type': 'scatter',
            'hoverinfo': 'text'
        }
        fig_network = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title='Ingredient Pairings and Your Food Identity',
            showlegend=False, hovermode='closest',
            margin=dict(b=10, l=5, r=5, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ))
        fig_network.update_layout(showlegend=False)
        network_description = "This simple network shows connections between 8 key ingredients, colored by cuisine (e.g., red for Italian/Mexican, green for Thai). Hover to see pairings like basil-oregano (Italian) or cumin-cilantro (Mexican), revealing your culinary identity."
        precomputed_visualizations['network'] = {
            'json': fig_network.to_json(),
            'description': network_description
        }

def format_recipe_title(title):
    """Convert a recipe title to Title Case, handling non-string inputs."""
    if isinstance(title, str):
        return title.title()
    return ""

def format_instructions(steps):
    """Format recipe steps into a numbered list for clarity."""
    if not steps or not isinstance(steps, list):
        return ""
    numbered_steps = [f"<li>{step.strip().capitalize()}</li>" for step in steps if step.strip()]
    return Markup("<ol>" + "".join(numbered_steps) + "</ol>") if numbered_steps else ""

def compute_match(user_ingredients, recipe_ingredients):
    """Compute match % efficiently."""
    user_set = set(i.lower().strip() for i in user_ingredients if i.strip())
    recipe_set = set(recipe_ingredients)
    if not recipe_set:
        return 0, set(), set()
    common = user_set.intersection(recipe_set)
    missing = recipe_set - user_set
    match_percent = (len(common) / len(recipe_set)) * 100 if recipe_set else 0
    return match_percent, common, missing

def suggest_substitutions(missing_ingredients):
    """Suggest substitutions quickly."""
    substitutions = {
        'basil': 'oregano', 'butter': 'olive oil', 'honey': 'maple syrup',
        'salt': 'soy sauce', 'pepper': 'chili flakes'
    }
    return {ing: subs for ing, subs in substitutions.items() if ing in missing_ingredients}

def compute_ingredients_stats(df):
    """Compute stats efficiently."""
    df = df.copy()
    df['num_ingredients'] = df['ingredients'].apply(len)
    df['instruction_length'] = df['steps'].apply(lambda steps: len(" ".join(steps).split()) if steps else 0)
    return df

def build_cooccurrence_network(df, top_n=15):
    """Build network with reduced complexity for performance (used in precompute)."""
    pass  # Already handled in precompute_visualizations

def infer_culinary_identity(user_ingredients, matched_recipes):
    """Infer identity efficiently."""
    ingredient_to_cuisine = {
        'basil': ['Italian', 'Thai'], 'oregano': ['Italian', 'Greek'], 'soy sauce': ['Japanese', 'Chinese', 'Korean'],
        'tortilla': ['Mexican'], 'ginger': ['Chinese', 'Thai', 'Indian'], 'cumin': ['Mexican', 'Indian', 'Middle Eastern'],
        'cilantro': ['Mexican', 'Thai', 'Indian'], 'parmesan': ['Italian'], 'sriracha': ['Thai'],
        'curry': ['Indian', 'Thai'], 'feta': ['Greek', 'Mediterranean'], 'paprika': ['Spanish', 'Hungarian'],
        'saffron': ['Spanish', 'Middle Eastern'], 'tahini': ['Middle East'], 'rice vinegar': ['Japanese', 'Chinese'],
        'coconut milk': ['Thai', 'Indian', 'Caribbean']
    }
    inferred = []
    for ing in user_ingredients:
        if ing in ingredient_to_cuisine:
            inferred.extend(ingredient_to_cuisine[ing])
    cuisine_votes = {}
    for recipe in matched_recipes:
        name_lower = recipe['recipe_name'].lower()
        tags = recipe.get('tags', [])
        for word in name_lower.split() + [tag.lower() for tag in tags]:
            if word in {'mexican', 'italian', 'thai', 'indian', 'chinese', 'japanese', 'greek', 'mediterranean', 'french', 'spanish', 'middle eastern', 'caribbean', 'korean'}:
                cuisine_votes[word.capitalize()] = cuisine_votes.get(word.capitalize(), 0) + 1
    return [max(cuisine_votes, key=cuisine_votes.get)] if cuisine_votes else ['General']

def infer_flavor_profile(user_ingredients):
    """Infer flavor profile efficiently, including 'General' profile."""
    flavor_to_ingredients = {
        'Savory': ['soy sauce', 'parmesan', 'salt', 'pepper'], 'Sweet': ['honey', 'sugar', 'maple syrup'],
        'Spicy': ['chili flakes', 'sriracha', 'ginger'], 'Herbaceous': ['basil', 'oregano', 'cilantro'],
        'Umami': ['mushrooms', 'tomatoes', 'tahini']
    }
    flavor_scores = {flavor: 0 for flavor in flavor_to_ingredients}
    for ing in user_ingredients:
        for flavor, ingredients in flavor_to_ingredients.items():
            if ing in ingredients:
                flavor_scores[flavor] += 1
    total_flavors = sum(flavor_scores.values())
    if total_flavors == 0 or all(score == 0 for score in flavor_scores.values()):
        return {flavor: 1 for flavor in flavor_to_ingredients}, "Your culinary style leans towards General. You appreciate a balanced mix of Savory, Sweet, Spicy, Herbaceous, and Umami flavors, suggesting a versatile food identity."
    dominant = max(flavor_scores, key=flavor_scores.get)
    top_flavors = sorted(flavor_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    return flavor_scores, f"Your culinary style leans towards {dominant}. You appreciate dishes with a balance of {top_flavors[0][0]} and {top_flavors[1][0]}."

def infer_ancestry(user_ingredients):
    """Infer ancestry percentage efficiently."""
    cuisine_to_ingredients = {
        'Italian': ['basil', 'oregano', 'parmesan'], 'Mexican': ['tortilla', 'cumin', 'cilantro'],
        'Thai': ['basil', 'ginger', 'sriracha'], 'Indian': ['cumin', 'ginger', 'curry'],
        'Chinese': ['soy sauce', 'ginger', 'rice vinegar'], 'Japanese': ['soy sauce', 'rice vinegar'],
        'Greek': ['feta', 'oregano'], 'Middle Eastern': ['tahini', 'saffron', 'cumin'],
        'Spanish': ['paprika', 'saffron'], 'Caribbean': ['coconut milk']
    }
    total = len(user_ingredients)
    if total == 0:
        return {'General': 100}, "Your culinary ancestry is General (100%)."
    cuisine_scores = {cuisine: 0 for cuisine in cuisine_to_ingredients}
    for ing in user_ingredients:
        for cuisine, ingredients in cuisine_to_ingredients.items():
            if ing in ingredients:
                cuisine_scores[cuisine] += 1
    total_matches = sum(cuisine_scores.values())
    if total_matches == 0:
        return {'General': 100}, "Your culinary ancestry is General (100%)."
    percentages = {c: (v / total_matches * 100) for c, v in cuisine_scores.items() if v > 0}
    dominant = max(percentages, key=percentages.get) if percentages else 'General'
    return percentages, f"You are {percentages[dominant]:.0f}% {dominant}. Explore recipes from this heritage!"

def get_ingredient_history(ingredient):
    """Fetch ingredient history using Wikipedia API (placeholder)."""
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={ingredient}_(food)&prop=extracts&format=json&exintro=1"
        headers = {'User-Agent': 'DeconstructionistCookbook/1.0 (your.email@example.com)'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page in pages.values():
                extract = page.get('extract', '')
                if extract:
                    return f"History of {ingredient}: {extract.split('.')[0]}. (More details on Wikipedia)"
        return f"History of {ingredient} not found. (Check Wikipedia for {ingredient}.)"
    except Exception as e:
        return f"Error fetching history for {ingredient}: {str(e)}"

def get_ingredient_origin(ingredient):
    """Fetch ingredient origin (placeholder, replace with map data or API)."""
    origins = {
        'basil': 'India/Mediterranean', 'oregano': 'Mediterranean', 'soy sauce': 'China/Japan',
        'tortilla': 'Mexico', 'ginger': 'Southeast Asia', 'cumin': 'Middle East/India',
        'cilantro': 'Mediterranean/Mexico', 'parmesan': 'Italy', 'sriracha': 'Thailand',
        'curry': 'India/Thailand', 'feta': 'Greece', 'paprika': 'Spain/Hungary',
        'saffron': 'Spain/Middle East', 'tahini': 'Middle East', 'rice vinegar': 'China/Japan',
        'coconut milk': 'Southeast Asia/Caribbean'
    }
    return origins.get(ingredient, 'Unknown origin')

@app.route('/')
def index():
    """Render landing page efficiently."""
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    """Process user input efficiently, showing only top 10 matching recipes with identity insights."""
    user_input = request.form.get('ingredients', '').strip()
    user_ingredients = [i.strip().lower() for i in user_input.split(',') if i.strip()]
    
    matched_recipes = []
    for idx, row in load_raw_recipes().iterrows():  # Use iterator for memory efficiency
        match_percent, have, missing = compute_match(user_ingredients, row['ingredients'])
        if match_percent >= 10:
            subs = suggest_substitutions(missing)
            matched_recipes.append({
                'recipe_name': format_recipe_title(row['name']),
                'match_percent': round(match_percent, 2),
                'recipe_ingredients': row['ingredients'],
                'have': list(have),
                'missing': list(missing),
                'substitutions': subs,
                'instructions': format_instructions(row['steps'] if row['steps'] else []),
                'source': 'RAW'
            })
    
    # Sort by match percentage in descending order and limit to top 10
    matched_recipes = sorted(matched_recipes, key=lambda x: x['match_percent'], reverse=True)[:10]
    
    culinary_identity = infer_culinary_identity(user_ingredients, matched_recipes)
    flavor_scores, flavor_message = infer_flavor_profile(user_ingredients)
    ancestry_percentages, ancestry_message = infer_ancestry(user_ingredients)
    
    # Simplified visualizations with hover only
    # Flavor Profile Radar Chart (integrated into Visuals tab)
    flavor_labels = list(flavor_scores.keys())
    flavor_values = list(flavor_scores.values())
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=flavor_values, theta=flavor_labels, fill='toself', name='Flavor Profile',
        hovertemplate='Flavor: %{theta}<br>Strength: %{r}<extra></extra>'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(flavor_values) + 1], showticklabels=False)),
        showlegend=False, title='Your Flavor Profile', template='plotly_white'
    )
    # Use to_json() for serialization, config handled in frontend
    graphJSON_radar = fig_radar.to_json()
    radar_description = "Hover over this radar to see your flavor strengths (e.g., Savory, Herbaceous), revealing your culinary identity. A high Herbaceous score might suggest Italian or Thai influences, while a balanced profile indicates a versatile, General food identity."
    
    # Use precomputed visualizations with descriptions
    return render_template('results.html',
                           recipes=matched_recipes,
                           user_ingredients=user_ingredients,
                           culinary_identity=culinary_identity,
                           flavor_message=flavor_message,
                           ancestry_message=ancestry_message,
                           graphJSON_freq=precomputed_visualizations['freq']['json'],
                           freq_description=precomputed_visualizations['freq']['description'],
                           graphJSON_hist=precomputed_visualizations['hist']['json'],
                           hist_description=precomputed_visualizations['hist']['description'],
                           graphJSON_scatter=precomputed_visualizations['scatter']['json'],
                           scatter_description=precomputed_visualizations['scatter']['description'],
                           graphJSON_network=precomputed_visualizations['network']['json'],
                           network_description=precomputed_visualizations['network']['description'],
                           graphJSON_radar=graphJSON_radar,
                           radar_description=radar_description)

# Identity pages
@app.route('/ancestry_report')
def ancestry_report():
    user_ingredients = request.args.get('ingredients', '').split(',')
    user_ingredients = [i.strip().lower() for i in user_ingredients if i.strip()]
    _, ancestry_message = infer_ancestry(user_ingredients)
    # Suggest recipes from dominant ancestry
    dominant_cuisine = ancestry_message.split(' ')[-1].rstrip('.')  # Extract cuisine (e.g., "Italian")
    ancestry_recipes = []
    if dominant_cuisine != 'General':
        for idx, row in load_raw_recipes().iterrows():
            name_lower = row['name'].lower()
            if dominant_cuisine.lower() in name_lower or any(tag.lower() == dominant_cuisine.lower() for tag in row['tags']):
                ancestry_recipes.append({
                    'recipe_name': format_recipe_title(row['name']),
                    'instructions': format_instructions(row['steps'] if row['steps'] else [])
                })
        ancestry_recipes = ancestry_recipes[:5]  # Top 5 recipes from this cuisine
    return render_template('ancestry_report.html', ancestry_message=ancestry_message, user_ingredients=user_ingredients, ancestry_recipes=ancestry_recipes)

@app.route('/recipe_time_travel')
def recipe_time_travel():
    user_ingredients = request.args.get('ingredients', '').split(',')
    user_ingredients = [i.strip().lower() for i in user_ingredients if i.strip()]
    history = []
    for ing in user_ingredients:
        history.append(get_ingredient_history(ing))
    return render_template('recipe_time_travel.html', history="\n".join(history), user_ingredients=user_ingredients)

@app.route('/ingredient_origins_map')
def ingredient_origins_map():
    user_ingredients = request.args.get('ingredients', '').split(',')
    user_ingredients = [i.strip().lower() for i in user_ingredients if i.strip()]
    origins = []
    origin_coords = []  # For mapping
    for ing in user_ingredients:
        origin = get_ingredient_origin(ing)
        origins.append(f"{ing}: {origin}")
        # Placeholder coordinates (replace with real data or API)
        if 'India' in origin: origin_coords.append([20.5937, 78.9629])  # India
        elif 'Mediterranean' in origin: origin_coords.append([35.1796, 24.3850])  # Mediterranean (approx.)
        elif 'Mexico' in origin: origin_coords.append([23.6345, -102.5528])  # Mexico
        elif 'China' in origin or 'Japan' in origin: origin_coords.append([35.8617, 104.1954])  # China/Japan (approx.)
        elif 'Southeast Asia' in origin: origin_coords.append([11.6098, 104.9915])  # Southeast Asia (approx.)
        elif 'Middle East' in origin: origin_coords.append([29.3117, 47.4818])  # Middle East (approx.)
        elif 'Spain' in origin: origin_coords.append([40.4637, -3.7492])  # Spain
        elif 'Greece' in origin: origin_coords.append([39.0742, 21.8243])  # Greece
        elif 'Caribbean' in origin: origin_coords.append([21.2747, -78.0186])  # Caribbean (approx.)
        else: origin_coords.append([0, 0])  # Default (unknown)
    return render_template('ingredient_origins_map.html', origins="\n".join(origins), user_ingredients=user_ingredients, origin_coords=origin_coords)

if __name__ == '__main__':
    precompute_visualizations()  # Precompute on startup
    load_raw_recipes()
    app.run(debug=False)