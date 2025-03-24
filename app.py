import ast
import os
from markupsafe import Markup
from flask import Flask, render_template, request, url_for, redirect, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import nltk
import plotly.graph_objects as go
import time
import json

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"
db = SQLAlchemy(app)

RAW_RECIPES_FILE = os.path.join('data', 'RAW_recipes.csv')
df_raw = None

class LikedRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.Integer, nullable=False)

class FoodMemory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(300))
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    voice_note_path = db.Column(db.String(300))
    year = db.Column(db.Integer)
    emotions = db.Column(db.String(200))
    friend_voices = db.Column(db.Text)  # Store as JSON string

with app.app_context():
    db.create_all()

def load_raw_recipes():
    global df_raw
    if df_raw is None:
        try:
            # Load more rows from the dataset to provide better recipe suggestions
            df_raw = pd.read_csv(RAW_RECIPES_FILE, nrows=5000)
            
            # Process columns that need to be parsed from string to list
            for col in ['tags', 'steps', 'ingredients']:
                if col in df_raw.columns:
                    df_raw[col] = df_raw[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
            
            # Convert ingredients to lowercase
            if 'ingredients' in df_raw.columns:
                df_raw['ingredients'] = df_raw['ingredients'].apply(lambda lst: [i.lower().strip() for i in lst])
                
            # Fill missing names
            df_raw['name'] = df_raw['name'].fillna('').astype(str)
            
        except Exception as e:
            print(f"Error loading recipes: {e}")
            # Create a minimal dataframe with sample data in case of error
            df_raw = pd.DataFrame({
                'id': range(1, 6),
                'name': ['Simple Pasta', 'Basic Salad', 'Veggie Stir Fry', 'Tomato Soup', 'Rice Bowl'],
                'ingredients': [
                    ['pasta', 'olive oil', 'garlic', 'salt'],
                    ['lettuce', 'tomato', 'cucumber', 'olive oil', 'vinegar'],
                    ['tofu', 'broccoli', 'carrots', 'soy sauce', 'ginger'],
                    ['tomatoes', 'onion', 'vegetable broth', 'basil'],
                    ['rice', 'beans', 'corn', 'cilantro', 'lime']
                ],
                'steps': [
                    ['Cook pasta', 'Sauté garlic in oil', 'Mix together', 'Season with salt'],
                    ['Chop vegetables', 'Mix in bowl', 'Dress with oil and vinegar'],
                    ['Stir fry vegetables', 'Add tofu', 'Season with soy sauce and ginger'],
                    ['Sauté onion', 'Add tomatoes', 'Add broth', 'Simmer', 'Garnish with basil'],
                    ['Cook rice', 'Mix with beans and corn', 'Top with cilantro and lime']
                ],
                'tags': [
                    ['Italian', 'Pasta', 'Quick'],
                    ['Salad', 'Healthy', 'Fresh'],
                    ['Asian', 'Vegetarian', 'Stir Fry'],
                    ['Soup', 'Tomato', 'Comfort Food'],
                    ['Mexican', 'Bowl', 'Vegetarian']
                ]
            })
            
    return df_raw

def format_recipe_title(title):
    return title.title() if isinstance(title, str) else ""

def format_instructions(steps):
    if not steps or not isinstance(steps, list):
        return ""
    numbered_steps = [f"<li>{step.strip().capitalize()}</li>" for step in steps if step.strip()]
    return Markup("<ol>" + "".join(numbered_steps) + "</ol>") if numbered_steps else ""

def compute_match(user_ingredients, recipe_ingredients):
    user_set = set(i.lower().strip() for i in user_ingredients if i.strip())
    recipe_set = set(recipe_ingredients)
    if not recipe_set:
        return 0, set(), set()
    common = user_set.intersection(recipe_set)
    missing = recipe_set - user_set
    match_percent = (len(common) / len(recipe_set)) * 100 if recipe_set else 0
    return match_percent, common, missing
   
def suggest_substitutions(missing_ingredients):
    substitutions = {
        'basil': 'oregano', 'butter': 'olive oil', 'honey': 'maple syrup',
        'salt': 'soy sauce', 'pepper': 'chili flakes'
    }
    return {ing: subs for ing, subs in substitutions.items() if ing in missing_ingredients}

def infer_culinary_identity(user_ingredients, matched_recipes):
    ingredient_to_cuisine = {
        'basil': ['Italian', 'Thai'], 'oregano': ['Italian', 'Greek'], 'soy sauce': ['Japanese', 'Chinese', 'Korean'],
        'tortilla': ['Mexican'], 'ginger': ['Chinese', 'Thai', 'Indian'], 'cumin': ['Mexican', 'Indian', 'Middle Eastern'],
        'cilantro': ['Mexican', 'Thai', 'Indian'], 'parmesan': ['Italian'], 'sriracha': ['Thai'],
        'curry': ['Indian', 'Thai'], 'feta': ['Greek', 'Mediterranean'], 'paprika': ['Spanish', 'Hungarian'],
        'saffron': ['Spanish', 'Middle Eastern'], 'tahini': ['Middle Eastern'], 'rice vinegar': ['Japanese', 'Chinese'],
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
    flavor_to_ingredients = {
        'Savory': ['soy sauce', 'parmesan', 'salt', 'pepper'],
        'Sweet': ['honey', 'sugar', 'maple syrup'],
        'Spicy': ['chili flakes', 'sriracha', 'ginger'],
        'Herbaceous': ['basil', 'oregano', 'cilantro'],
        'Umami': ['mushrooms', 'tomatoes', 'tahini']
    }
    flavor_scores = {flavor: 0 for flavor in flavor_to_ingredients}
    for ing in user_ingredients:
        for flavor, ingredients in flavor_to_ingredients.items():
            if ing in ingredients:
                flavor_scores[flavor] += 1
    total = sum(flavor_scores.values())
    if total == 0 or all(score == 0 for score in flavor_scores.values()):
        return {flavor: 1 for flavor in flavor_to_ingredients}, "Your culinary style is a balanced mix of flavors!"
    dominant = max(flavor_scores, key=flavor_scores.get)
    top_flavors = sorted(flavor_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    return flavor_scores, f"Your palate dances with {dominant} notes, blending {top_flavors[0][0]} and {top_flavors[1][0]}!"

def infer_ancestry(user_ingredients):
    cuisine_to_ingredients = {
        'Italian': ['basil', 'oregano', 'parmesan'],
        'Mexican': ['tortilla', 'cumin', 'cilantro'],
        'Thai': ['basil', 'ginger', 'sriracha'],
        'Indian': ['cumin', 'ginger', 'curry'],
        'Chinese': ['soy sauce', 'ginger', 'rice vinegar'],
        'Japanese': ['soy sauce', 'rice vinegar'],
        'Greek': ['feta', 'oregano'],
        'Middle Eastern': ['tahini', 'saffron', 'cumin'],
        'Spanish': ['paprika', 'saffron'],
        'Caribbean': ['coconut milk']
    }
    if not user_ingredients:
        return {}, "No ingredients to trace your culinary roots!", "General"
    cuisine_scores = {cuisine: 0 for cuisine in cuisine_to_ingredients}
    for ing in user_ingredients:
        for cuisine, ings in cuisine_to_ingredients.items():
            if ing in ings:
                cuisine_scores[cuisine] += 1
    if sum(cuisine_scores.values()) == 0:
        return {}, "Your ingredients hint at a unique, uncharted culinary story!", "General"
    dominant = max(cuisine_scores, key=cuisine_scores.get)
    message = f"Your ingredients whisper tales of {dominant} heritage—savor its flavors!"
    return cuisine_scores, message, dominant

@app.route('/')
def landing():
    return render_template('index.html')

# Add alias route for index to prevent URL build errors
@app.route('/index')
def index():
    return redirect(url_for('landing'))

@app.route('/cookbook')
def cookbook():
    return render_template('cookbook.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        user_input = request.form.get('ingredients', '').strip()
    else:
        user_input = request.args.get('ingredients', '').strip()
        
    # Handle comma-separated ingredients from the textarea
    user_ingredients = [i.strip().lower() for i in user_input.split(',') if i.strip()]
    
    if not user_ingredients:
        return redirect(url_for('cookbook'))

    try:
        matched_recipes = []
        recipes_df = load_raw_recipes()
        
        # Process all available recipes
        for idx, row in recipes_df.iterrows():
            # No limit on processing - process all available recipes
            match_percent, have, missing = compute_match(user_ingredients, row['ingredients'])
            if match_percent >= 10:
                subs = suggest_substitutions(missing)
                matched_recipes.append({
                    'id': row['id'] if 'id' in row else idx,
                    'recipe_name': format_recipe_title(row['name']) if 'name' in row else f"Recipe {idx}",
                    'match_percent': round(match_percent, 2),
                    'recipe_ingredients': row['ingredients'] if 'ingredients' in row else [],
                    'have': list(have),
                    'missing': list(missing),
                    'substitutions': subs,
                    'instructions': format_instructions(row['steps'] if 'steps' in row else []),
                    'tags': row.get('tags', [])
                })
                
        # Sort and limit to more recipes (10 instead of 5)
        matched_recipes = sorted(matched_recipes, key=lambda x: x['match_percent'], reverse=True)[:10]

        # Generate culinary analysis
        culinary_identity = infer_culinary_identity(user_ingredients, matched_recipes)
        flavor_scores, flavor_message = infer_flavor_profile(user_ingredients)
        ancestry_scores, ancestry_message, dominant = infer_ancestry(user_ingredients)

        # Flavor Profile Radar Chart
        flavor_labels = list(flavor_scores.keys())
        flavor_values = list(flavor_scores.values())
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=flavor_values, theta=flavor_labels, fill='toself', name='Flavor Profile',
            hovertemplate='Flavor: %{theta}<br>Strength: %{r}<extra></extra>'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(flavor_values) + 2 if flavor_values else 3])),
            showlegend=False, title='Your Flavor Palette', template='plotly_white'
        )
        graphJSON_radar = fig_radar.to_json()

        # Ancestry Bar Chart
        fig_ancestry = go.Figure([go.Bar(x=list(ancestry_scores.keys()), y=list(ancestry_scores.values()), marker_color="#DAA520")])
        fig_ancestry.update_layout(title='Your Culinary Roots', template='plotly_white')
        graphJSON_ancestry = fig_ancestry.to_json()

        # Liked Recipes and Identity Visualization (simplified)
        graphJSON_liked_tags = None

        return render_template('results.html',
                            recipes=matched_recipes,
                            liked_recipes=[],  # Simplified
                            user_ingredients=user_ingredients,
                            culinary_identity=culinary_identity,
                            flavor_message=flavor_message,
                            ancestry_message=ancestry_message,
                            graphJSON_radar=graphJSON_radar,
                            graphJSON_ancestry=graphJSON_ancestry,
                            graphJSON_liked_tags=graphJSON_liked_tags)
                            
    except Exception as e:
        # Error handling with fallback content
        print(f"Error in results route: {e}")
        
        # Create default visualizations for error case
        default_radar = go.Figure(data=go.Scatterpolar(
            r=[1, 1, 1, 1, 1], 
            theta=['Savory', 'Sweet', 'Spicy', 'Herbaceous', 'Umami'], 
            fill='toself'
        ))
        default_radar.update_layout(title='Flavor Profile', template='plotly_white')
        
        return render_template('results.html',
                            recipes=[],
                            user_ingredients=user_ingredients,
                            culinary_identity=['General'],
                            flavor_message="We couldn't process your ingredients at this time.",
                            ancestry_message="Try again with different ingredients.",
                            graphJSON_radar=default_radar.to_json(),
                            graphJSON_ancestry=default_radar.to_json(),
                            graphJSON_liked_tags=None)

@app.route('/like_recipe', methods=['POST'])
def like_recipe():
    recipe_id = request.form.get('recipe_id')
    ingredients = request.form.get('ingredients', '')
    if recipe_id:
        like = LikedRecipe(recipe_id=int(recipe_id))
        db.session.add(like)
        db.session.commit()
    return redirect(url_for('results', ingredients=ingredients))

@app.route('/ancestry_report')
def ancestry_report():
    try:
        ingredients_param = request.args.get('ingredients', '')
        if not ingredients_param:
            return render_template('ancestry_report.html',
                               ancestry_message="Please enter ingredients to uncover your culinary roots!",
                               user_ingredients=[], ancestry_recipes=[])
        
        user_ingredients = [i.strip().lower() for i in ingredients_param.split(',') if i.strip()]
        if not user_ingredients:
            return render_template('ancestry_report.html',
                               ancestry_message="Please enter valid ingredients to uncover your culinary roots!",
                               user_ingredients=[], ancestry_recipes=[])
        
        _, ancestry_message, dominant = infer_ancestry(user_ingredients)
        ancestry_recipes = []
        
        if dominant != 'General':
            raw_recipes = load_raw_recipes()
            for idx, row in raw_recipes.iterrows():
                name_lower = row['name'].lower() if isinstance(row['name'], str) else ''
                tags = row.get('tags', []) if isinstance(row.get('tags'), list) else []
                
                if dominant.lower() in name_lower or any(tag.lower() == dominant.lower() for tag in tags):
                    ancestry_recipes.append({
                        'recipe_name': format_recipe_title(row['name']),
                        'instructions': format_instructions(row['steps'] if row['steps'] else [])
                    })
            ancestry_recipes = ancestry_recipes[:5]
        
        return render_template('ancestry_report.html',
                           ancestry_message=ancestry_message,
                           user_ingredients=user_ingredients,
                           ancestry_recipes=ancestry_recipes)
    except Exception as e:
        print(f"Error in ancestry_report: {str(e)}")
        return render_template('ancestry_report.html',
                           ancestry_message="An error occurred while processing your request. Please try again.",
                           user_ingredients=[], ancestry_recipes=[])

@app.route('/food_memory_diary', methods=['GET', 'POST'])
def food_memory_diary():
    memories = FoodMemory.query.order_by(FoodMemory.created_at.desc()).all()
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        year = request.form.get('year')
        emotions = request.form.get('emotions')
        
        # Handle image upload
        image_path = None
        if 'image' in request.files:
            image = request.files['image']
            if image and image.filename:
                filename = f"memory_{int(time.time())}_{image.filename}"
                image_path = os.path.join('static', 'uploads', filename)
                image.save(os.path.join(app.root_path, image_path))
        
        # Handle voice note upload
        voice_note_path = None
        if 'voice_note' in request.files:
            voice_note = request.files['voice_note']
            if voice_note and voice_note.filename:
                filename = f"voice_{int(time.time())}_{voice_note.filename}"
                voice_note_path = os.path.join('static', 'uploads', filename)
                voice_note.save(os.path.join(app.root_path, voice_note_path))
        
        # Handle friend voices
        friend_voices = []
        for i in range(1, 4):  # Support up to 3 friend voices
            friend_name = request.form.get(f'friend_name_{i}')
            friend_voice = request.files.get(f'friend_voice_{i}')
            if friend_name and friend_voice:
                filename = f"friend_{int(time.time())}_{i}_{friend_voice.filename}"
                voice_path = os.path.join('static', 'uploads', filename)
                friend_voice.save(os.path.join(app.root_path, voice_path))
                friend_voices.append({
                    'name': friend_name,
                    'voice_path': voice_path
                })
        
        memory = FoodMemory(
            name=name,
            description=description,
            image_path=image_path,
            voice_note_path=voice_note_path,
            latitude=latitude,
            longitude=longitude,
            year=year,
            emotions=emotions,
            friend_voices=json.dumps(friend_voices) if friend_voices else None
        )
        
        db.session.add(memory)
        db.session.commit()
        
        flash('Food memory saved successfully!', 'success')
        return redirect(url_for('food_memory_map'))
    
    return render_template('food_memory_diary.html', memories=memories)

@app.route('/food_memory_map')
def food_memory_map():
    memories = FoodMemory.query.order_by(FoodMemory.created_at.desc()).all()
    memory_list = []
    for mem in memories:
        memory_data = {
            'id': mem.id,
            'name': mem.name,
            'description': mem.description,
            'image_path': mem.image_path,
            'voice_note_path': mem.voice_note_path,
            'latitude': mem.latitude,
            'longitude': mem.longitude,
            'year': mem.year,
            'emotions': mem.emotions,
            'created_at': mem.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'friend_voices': json.loads(mem.friend_voices) if mem.friend_voices else []
        }
        memory_list.append(memory_data)
    return render_template('food_memory_map.html', memories=memory_list)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/flavor_profile')
def flavor_profile():
    ingredients_param = request.args.get('ingredients', '')
    user_ingredients = [i.strip().lower() for i in ingredients_param.split(',') if i.strip()]
    
    # Generate flavor profile data
    flavor_scores, flavor_message = infer_flavor_profile(user_ingredients)
    
    # Create radar chart for flavor profile
    flavor_labels = list(flavor_scores.keys())
    flavor_values = list(flavor_scores.values())
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=flavor_values, theta=flavor_labels, fill='toself', name='Flavor Profile',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255, 107, 107, 0.6)',
        hovertemplate='Flavor: %{theta}<br>Strength: %{r}<extra></extra>'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(flavor_values) + 2 if flavor_values else 3]),
            bgcolor='rgba(255, 248, 248, 0.3)'
        ),
        showlegend=False, 
        title={
            'text': 'Your Flavor Palette',
            'font': {'size': 24, 'color': '#333'}
        },
        template='plotly_white',
        paper_bgcolor='rgba(255, 255, 255, 0.8)',
        plot_bgcolor='rgba(255, 255, 255, 0)',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    graphJSON_radar = fig_radar.to_json()
    
    # Get cuisine and ancestry data
    cuisine_scores, _, dominant_cuisine = infer_ancestry(user_ingredients)
    
    # Create Culinary Heritage Tree Map
    labels = list(cuisine_scores.keys())
    values = list(cuisine_scores.values())
    
    if sum(values) == 0:
        # If no cuisine matches, provide a default visualization
        labels = ['Unique Style']
        values = [1]
    
    fig_heritage = go.Figure(go.Treemap(
        labels=labels,
        values=values,
        parents=[''] * len(labels),
        textinfo='label+value',
        hoverinfo='label+value+percent parent',
        marker=dict(
            colors=['#FF6B6B', '#FF8E53', '#FFBD80', '#FFD1A9', '#FFDEC2'],
            line=dict(width=2, color='white')
        ),
    ))
    
    fig_heritage.update_layout(
        title={
            'text': 'Your Culinary Heritage',
            'font': {'size': 24, 'color': '#333'}
        },
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(255, 255, 255, 0.8)',
    )
    
    graphJSON_heritage = fig_heritage.to_json()
    
    return render_template('flavor_profile.html',
                          user_ingredients=user_ingredients,
                          flavor_message=flavor_message,
                          graphJSON_radar=graphJSON_radar,
                          graphJSON_heritage=graphJSON_heritage,
                          dominant_cuisine=dominant_cuisine if 'dominant_cuisine' in locals() else 'Unique')

if __name__ == '__main__':
    load_raw_recipes()
    app.run(debug=False)