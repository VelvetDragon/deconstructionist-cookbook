# Deconstructionist Cookbook

A modern, interactive culinary application that analyzes food memories, cooking styles, and flavor profiles to create a personalized cooking experience.

## Features

- **Food Memory Map**: Record and visualize your food memories geographically
- **Flavor Profile Analysis**: Discover your unique culinary identity and flavor preferences
- **Interactive Recipe Suggestions**: Get personalized recipe recommendations based on your flavor profile
- **Voice Recording**: Capture food memories with both text and voice recordings
- **Friend Contributions**: Allow friends to add their memories to your food map pins

## Deployment

This application is ready to be deployed using free hosting services like Render or PythonAnywhere.

### Deploying on Render

1. Create a Render account at [render.com](https://render.com)
2. Connect your GitHub repository with this code
3. Create a new Web Service with the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
   - Environment: Python 3.9 (or later)
4. Your app will be live at your Render URL when the build completes

### Deploying on PythonAnywhere

1. Create a PythonAnywhere account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your code or clone your repository
3. Set up a Web app using Flask and point it to your wsgi.py file
4. Make sure to create a virtual environment and install all packages from requirements.txt
5. Your app will be live at yourusername.pythonanywhere.com

## Local Development

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Run the application: `python app.py`
6. Visit http://127.0.0.1:5000 in your browser

## Contributing Food Memories

If you want to contribute your food memories to this application, please share:

1. Name of the food/dish
2. Location (city, country, or coordinates)
3. Brief description of your memory related to this food
4. Optional: Image of the food
5. Optional: Voice recording describing your memory

## License

This project is created for educational purposes. Feel free to use, modify, and share. 