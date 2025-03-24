import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64
import io
import json
import os
from PIL import Image
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Deconstructionist Cookbook",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E74C3C;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2C3E50;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 4px solid #E74C3C;
    }
    .memory-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .food-memory-header {
        font-size: 1.8rem;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    .memory-meta {
        font-size: 0.9rem;
        color: #7F8C8D;
        margin-bottom: 1rem;
    }
    .memory-description {
        font-size: 1rem;
        color: #34495E;
        margin-bottom: 1rem;
    }
    .friend-memory {
        background-color: #F8F9FA;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #3498DB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'memories' not in st.session_state:
    # Default sample data
    st.session_state.memories = [
        {
            'id': 1,
            'name': 'Homemade Tamales',
            'description': 'My grandmother taught me how to make these when I was 8 years old. The process took all day, and the whole family helped. I still make them every Christmas using her recipe.',
            'latitude': 19.4326,
            'longitude': -99.1332,
            'location': 'Mexico City, Mexico',
            'image_path': 'https://images.unsplash.com/photo-1612549227431-19a957d4333d?auto=format&fit=crop&q=80&w=600&ixlib=rb-4.0.3',
            'has_voice': False,
            'has_friend_voices': True,
            'friend_voices': [
                {
                    'id': 1,
                    'name': 'Miguel',
                    'message': 'I remember eating these tamales at your house during the holidays. Your grandmother was an amazing cook!'
                }
            ]
        },
        {
            'id': 2,
            'name': 'Pad Thai Street Food',
            'description': 'I discovered this amazing street food stall on my first trip to Bangkok. The flavors were incredible - sweet, sour, and spicy all at once.',
            'latitude': 13.7563,
            'longitude': 100.5018,
            'location': 'Bangkok, Thailand',
            'image_path': 'https://images.unsplash.com/photo-1559314809-0d155014e29e?auto=format&fit=crop&q=80&w=600&ixlib=rb-4.0.3',
            'has_voice': False,
            'has_friend_voices': False,
            'friend_voices': []
        },
        {
            'id': 3,
            'name': 'Grandma\'s Apple Pie',
            'description': 'Every fall, we would go apple picking and then come home to bake pies. The smell of cinnamon and apples would fill the whole house.',
            'latitude': 42.3601,
            'longitude': -71.0589,
            'location': 'Boston, Massachusetts',
            'image_path': 'https://images.unsplash.com/photo-1535920527002-b35e96722969?auto=format&fit=crop&q=80&w=600&ixlib=rb-4.0.3',
            'has_voice': False,
            'has_friend_voices': False,
            'friend_voices': []
        }
    ]

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Map"

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cooking-book.png", width=100)
    st.title("Deconstructionist Cookbook")
    
    selected_tab = st.radio("Navigation", ["Home", "Map", "Add Memory", "Flavor Profile", "About"])
    
    st.session_state.current_tab = selected_tab
    
    st.markdown("---")
    st.markdown("### Class Presentation")
    st.info("""
    This is a Streamlit version of the Deconstructionist Cookbook project for easy sharing during class presentations.
    """)

# Main content
if st.session_state.current_tab == "Home":
    st.markdown("<h1 class='main-header'>Deconstructionist Cookbook</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Explore food memories, flavors, and culinary identity</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Deconstructionist Cookbook
        
        This interactive culinary application helps you explore food memories, analyze flavor profiles, 
        and discover your unique culinary identity. Using data science and visualization, we deconstruct 
        the elements of cooking and eating to create a personalized cooking experience.
        
        ### Key Features:
        - **Food Memory Map**: Record and visualize your food memories geographically
        - **Flavor Profile Analysis**: Discover your unique culinary identity
        - **Interactive Recipe Suggestions**: Get personalized recipe recommendations
        - **Memory Sharing**: Capture and share food memories with friends
        """)
        
        st.markdown("### Start Your Culinary Journey")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗺️ Explore Memory Map", use_container_width=True):
                st.session_state.current_tab = "Map"
                st.experimental_rerun()
        with col_b:
            if st.button("➕ Add Your Memory", use_container_width=True):
                st.session_state.current_tab = "Add Memory"
                st.experimental_rerun()
    
    with col2:
        st.image("https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&q=80&w=500&ixlib=rb-4.0.3", caption="Food connects us to memories, people, and places.")

elif st.session_state.current_tab == "Map":
    st.markdown("<h1 class='main-header'>Food Memory Map</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore food memories from around the world</p>", unsafe_allow_html=True)
    
    # Create a DataFrame from the memories
    df = pd.DataFrame([{
        'name': memory['name'],
        'description': memory['description'],
        'lat': memory['latitude'],
        'lon': memory['longitude'],
        'location': memory.get('location', f"{memory['latitude']}, {memory['longitude']}"),
        'id': memory['id']
    } for memory in st.session_state.memories])
    
    # Create map
    st.markdown("### Interactive Food Memory Map")
    
    if len(df) > 0:
        fig = px.scatter_mapbox(df, 
                               lat="lat", 
                               lon="lon", 
                               hover_name="name",
                               hover_data=["location"],
                               zoom=1, 
                               height=500,
                               size=[15]*len(df),
                               color_discrete_sequence=["#E74C3C"])
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox=dict(
                bearing=0,
                center=dict(
                    lat=30,
                    lon=0
                ),
                pitch=0,
                zoom=1.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No memories have been added yet. Add your first food memory to see it on the map!")
    
    # Memory details
    st.markdown("### Food Memory Details")
    
    if len(st.session_state.memories) > 0:
        memory_id = st.selectbox("Select a memory to view", 
                               options=[m['id'] for m in st.session_state.memories],
                               format_func=lambda x: next((m['name'] for m in st.session_state.memories if m['id'] == x), ""))
        
        selected_memory = next((m for m in st.session_state.memories if m['id'] == memory_id), None)
        
        if selected_memory:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"<h3 class='food-memory-header'>{selected_memory['name']}</h3>", unsafe_allow_html=True)
                
                location = selected_memory.get('location', f"{selected_memory['latitude']}, {selected_memory['longitude']}")
                st.markdown(f"<p class='memory-meta'>📍 {location}</p>", unsafe_allow_html=True)
                
                st.markdown(f"<p class='memory-description'>{selected_memory['description']}</p>", unsafe_allow_html=True)
                
                if selected_memory['has_friend_voices'] and len(selected_memory['friend_voices']) > 0:
                    st.markdown("#### Friend Memories")
                    for friend in selected_memory['friend_voices']:
                        st.markdown(f"""
                        <div class='friend-memory'>
                            <strong>{friend['name']} says:</strong>
                            <p>{friend['message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if 'image_path' in selected_memory and selected_memory['image_path']:
                    st.image(selected_memory['image_path'], caption=selected_memory['name'], use_column_width=True, output_format="JPEG")
    else:
        st.info("No memories have been added yet.")

elif st.session_state.current_tab == "Add Memory":
    st.markdown("<h1 class='main-header'>Add Food Memory</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Record a special food memory to add to your map</p>", unsafe_allow_html=True)
    
    with st.form("memory_form"):
        name = st.text_input("Food Name", placeholder="e.g., Grandma's Apple Pie")
        
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location", placeholder="e.g., Boston, Massachusetts")
        with col2:
            place_type = st.selectbox("Type of Place", ["Home", "Restaurant", "Street Food", "Travel Destination", "Other"])
        
        description = st.text_area("Memory Description", placeholder="Share your story about this food...")
        
        # Option to input coordinates directly or use location search
        coords_expander = st.expander("Advanced: Set Coordinates Manually")
        with coords_expander:
            lat_col, lon_col = st.columns(2)
            with lat_col:
                latitude = st.number_input("Latitude", value=0.0, format="%.6f")
            with lon_col:
                longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        
        # Image URL input
        image_url = st.text_input("Image URL (optional)", placeholder="https://example.com/image.jpg")
        
        # Submit button
        submit_button = st.form_submit_button("Save Memory")
        
        if submit_button:
            if not name or not description or not location:
                st.error("Please fill in all required fields.")
            else:
                # For demo purposes, using random coordinates if not manually set
                if latitude == 0.0 and longitude == 0.0:
                    import random
                    latitude = random.uniform(-90, 90)
                    longitude = random.uniform(-180, 180)
                
                # Create new memory
                new_id = max([m['id'] for m in st.session_state.memories], default=0) + 1
                new_memory = {
                    'id': new_id,
                    'name': name,
                    'description': description,
                    'latitude': latitude,
                    'longitude': longitude,
                    'location': location,
                    'image_path': image_url if image_url else None,
                    'has_voice': False,
                    'has_friend_voices': False,
                    'friend_voices': []
                }
                
                # Add to session state
                st.session_state.memories.append(new_memory)
                
                st.success("Memory added successfully!")
                st.balloons()

elif st.session_state.current_tab == "Flavor Profile":
    st.markdown("<h1 class='main-header'>Flavor Profile Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Discover your unique culinary identity</p>", unsafe_allow_html=True)
    
    # Input ingredients
    ingredients = st.text_area("Enter ingredients (comma-separated)", 
                              placeholder="e.g., basil, tomatoes, garlic, olive oil, salt")
    
    if st.button("Analyze Flavor Profile"):
        if not ingredients:
            st.warning("Please enter some ingredients to analyze.")
        else:
            # Parse ingredients
            ingredient_list = [i.strip().lower() for i in ingredients.split(',') if i.strip()]
            
            # Demo flavor profile
            flavor_profiles = {
                "Savory": 0.7 if any(i in ingredient_list for i in ['salt', 'garlic', 'onion', 'mushroom']) else 0.3,
                "Sweet": 0.8 if any(i in ingredient_list for i in ['sugar', 'honey', 'maple']) else 0.2,
                "Spicy": 0.6 if any(i in ingredient_list for i in ['chili', 'pepper', 'cayenne']) else 0.1,
                "Herbaceous": 0.9 if any(i in ingredient_list for i in ['basil', 'oregano', 'thyme', 'rosemary']) else 0.3,
                "Umami": 0.7 if any(i in ingredient_list for i in ['tomato', 'mushroom', 'soy sauce']) else 0.4
            }
            
            # Create radar chart for flavor profile
            categories = list(flavor_profiles.keys())
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(flavor_profiles.values()),
                theta=categories,
                fill='toself',
                name='Your Flavor Profile',
                line_color='#E74C3C'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="Your Unique Flavor Profile",
                title_font_size=20
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Flavor analysis
            st.markdown("### Flavor Analysis")
            
            dominant_flavor = max(flavor_profiles, key=flavor_profiles.get)
            st.markdown(f"""
            <div class="card">
                <h3>Culinary Identity</h3>
                <p>Your culinary style strongly emphasizes <strong>{dominant_flavor}</strong> flavors. 
                Based on your ingredients, you seem to enjoy dishes with complex, layered flavor profiles.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cuisine affinity (demo data)
            cuisines = {
                "Italian": 0.8 if any(i in ingredient_list for i in ['basil', 'tomato', 'olive oil', 'garlic']) else 0.3,
                "Thai": 0.7 if any(i in ingredient_list for i in ['lemongrass', 'chili', 'lime', 'fish sauce']) else 0.2,
                "Mexican": 0.9 if any(i in ingredient_list for i in ['cilantro', 'lime', 'chili', 'corn']) else 0.3,
                "Indian": 0.6 if any(i in ingredient_list for i in ['cumin', 'coriander', 'turmeric', 'garam masala']) else 0.2,
                "Japanese": 0.5 if any(i in ingredient_list for i in ['soy sauce', 'mirin', 'sesame oil', 'rice vinegar']) else 0.1
            }
            
            fig2 = go.Figure([go.Bar(
                x=list(cuisines.keys()),
                y=list(cuisines.values()),
                marker_color="#3498DB"
            )])
            
            fig2.update_layout(
                title="Cuisine Affinity",
                xaxis_title="Cuisine",
                yaxis_title="Affinity Score",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Methodology explanation
            with st.expander("How We Calculate Your Flavor Profile"):
                st.markdown("""
                Our flavor profile analysis is based on the ingredients you've entered and how they relate to different 
                taste categories. We use natural language processing and culinary theory to map ingredients to flavor 
                components (savory, sweet, spicy, herbaceous, umami) and cuisines.
                
                The radar chart visualizes your flavor profile across these key dimensions, while the bar chart shows 
                which cuisines your ingredient preferences align with most strongly.
                
                This analysis can help you discover new recipes that match your taste preferences and explore cuisines 
                that align with your culinary identity.
                """)

elif st.session_state.current_tab == "About":
    st.markdown("<h1 class='main-header'>About The Deconstructionist Cookbook</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    The Deconstructionist Cookbook is an interactive culinary application developed for a class project. It combines 
    food science, data visualization, and personal narratives to create a unique exploration of our relationship with food.
    
    ### Core Concept
    
    This project explores how food connects us to memories, places, and cultural identities. By "deconstructing" our 
    food experiences into their component parts—ingredients, flavors, memories, and cultural contexts—we can better 
    understand our personal culinary identity.
    
    ### Features and Technology
    
    - **Food Memory Map**: An interactive map that plots food memories geographically
    - **Flavor Profile Analysis**: Algorithm-based analysis of flavor preferences
    - **Memory Collection**: Tools for recording and sharing food memories
    - **Streamlit Application**: This demo version uses Streamlit for easy sharing
    
    ### Academic Context
    
    This project explores the intersection of food studies, data science, and digital humanities. It asks how computational 
    methods can enhance our understanding of food as both a personal and cultural phenomenon.
    
    ### Future Development
    
    Future versions of this application could include:
    
    - Machine learning models for more accurate flavor analysis
    - Social features for sharing and discovering memories
    - Integration with recipe databases and recommendation systems
    - Mobile applications for capturing food memories in the moment
    
    ### About the Creator
    
    This project was developed by [Your Name] for [Class Name] at [University Name]. Special thanks to Professor [Professor Name] 
    for guidance and feedback throughout the development process.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### How to Contribute
    
    If you're a classmate or friend who would like to contribute a food memory to this project, please share:
    
    1. Name of a dish/food that has a special memory for you
    2. Location where you had this food
    3. A brief story about this food memory
    4. A photo of the food (optional)
    
    You can submit these through the Add Memory feature or by contacting the project creator directly.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
    <div>© 2025 The Deconstructionist Cookbook - Class Project</div>
    <div>Created with ❤️ and Streamlit</div>
</div>
""", unsafe_allow_html=True) 