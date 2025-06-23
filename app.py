import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import requests
import openai
from dotenv import load_dotenv
import os
import ollama
from functools import lru_cache
from translate import Translator
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Chat, ChatSession
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ayurmate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize NLTK and download required data
try:
    # Set NLTK data path to a local directory
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)

    # Try to load the data, download if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    load_dotenv()
    logger.info("Successfully initialized NLTK and environment")
except Exception as e:
    logger.error(f"Error initializing NLTK: {str(e)}")
    raise

# API keys from .env with validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

if not all([GOOGLE_API_KEY, SEARCH_ENGINE_ID, OPENAI_API_KEY]):
    logger.warning("Some API keys are missing from .env file")

try:
    # Load model and data
    logger.info("Loading model and data files...")
    model = load_model('model.h5', compile=False)  # Skip compilation for faster loading
    with open('data.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    with open('texts.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        classes = pickle.load(f)
    logger.info("Successfully loaded model and data files")
except Exception as e:
    logger.error(f"Error loading model or data files: {str(e)}")
    raise

# Cache for known ingredients and images
known_ingredients = set(
    r.lower() for intent in intents['intents'] 
    for r in intent.get('remedy', []) 
    if r.lower() != "unknown"
)
image_cache = {}
logger.info(f"Loaded {len(known_ingredients)} known ingredients")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            flash('No account found with this email. Please register first.', 'error')
            return redirect(url_for('login'))
        
        if not user.check_password(password):
            flash('Incorrect password. Please try again.', 'error')
            return redirect(url_for('login'))
        
        login_user(user)
        return redirect(url_for('chat'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login instead.', 'error')
            return redirect(url_for('login'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/get_chat_sessions')
@login_required
def get_chat_sessions():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated_at.desc()).all()
    return jsonify([{
        'id': session.id,
        'title': session.title,
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat(),
        'preview': session.messages[-1].message if session.messages else None
    } for session in sessions])

@app.route('/get_chat_session/<int:session_id>')
@login_required
def get_chat_session(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    return jsonify({
        'id': session.id,
        'title': session.title,
        'messages': [{
            'id': msg.id,
            'message': msg.message,
            'response': msg.response,
            'timestamp': msg.timestamp.isoformat(),
            'language': msg.language,
            'images': msg.images,
            'remedy_names': msg.remedy_names
        } for msg in session.messages]
    })

@app.route('/delete_chat_session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_chat_session(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(session)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/create_chat_session', methods=['POST'])
@login_required
def create_chat_session():
    data = request.get_json()
    session = ChatSession(
        user_id=current_user.id,
        title=data.get('title', 'New Chat')
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({'session_id': session.id})

@app.route('/get_bot_response', methods=['POST'])
@login_required
def get_bot_response():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        selected_language = data.get('language', 'en')
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        
        # Store original text for response translation
        original_text = user_message
        
        # Translate user query to English if not already in English
        if selected_language != 'en':
            user_message = translate_text(user_message, 'en')

        # Get bot response using the existing chatbot logic
        response_text, remedies = chatbot_response(user_message)
        
        # Get images for remedies
        image_urls = []
        remedy_names = []
        if "unknown" not in remedies:
            for remedy in remedies:
                images = get_remedy_with_image(remedy)
                if images:
                    image_urls.append(images[0])
                    formatted_name = remedy.replace('_', ' ').title()
                    remedy_names.append(formatted_name)
        
        # Translate response back to user's language if not English
        if selected_language != 'en':
            response_text = translate_text(response_text, selected_language)
        
        # Save the chat to database with images and remedy names
        chat = Chat(
            session_id=session_id,
            user_id=current_user.id,
            message=original_text,
            response=response_text,
            language=selected_language,
            timestamp=datetime.utcnow(),
            images=json.dumps(image_urls),
            remedy_names=json.dumps(remedy_names)
        )
        db.session.add(chat)
        
        # Update session title if it's the first message
        if len(session.messages) == 0:
            session.title = original_text[:50] + ('...' if len(original_text) > 50 else '')
        
        session.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "text": response_text,
            "images": image_urls,
            "remedyNames": remedy_names
        })
    except Exception as e:
        logger.error(f"Error in get_bot_response: {str(e)}")
        return jsonify({
            "text": "I apologize, but I encountered an error. Please try asking your health-related question again.",
            "images": [],
            "remedyNames": []
        })

@app.route("/translate", methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text')
        target_language = data.get('target_language')
        
        if not text or not target_language:
            return jsonify({'error': 'Missing text or target language'}), 400
            
        # Create translator for the target language
        translator = Translator(to_lang=target_language)
        
        # Split text into smaller chunks if it's too long
        max_chunk_size = 500
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                logger.error(f"Error translating chunk: {str(e)}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        # Combine translated chunks
        translated_text = ' '.join(translated_chunks)
        
        return jsonify({
            'translated_text': translated_text
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500

def get_ayurvedic_image(query):
    """Fetch Ayurvedic image using Google Custom Search API with caching."""
    if query in image_cache:
        return image_cache[query]

    try:
        logger.debug(f"Fetching image for query: {query}")
        url = f"https://www.googleapis.com/customsearch/v1"
        
        # Enhance search query to focus on natural ingredients
        search_terms = {
            'ginger': 'fresh ginger root',
            'tulsi': 'fresh tulsi leaves plant',
            'honey': 'pure natural honey comb',
            'turmeric': 'fresh turmeric root',
            'cinnamon': 'natural cinnamon bark',
            'pepper': 'black pepper seeds',
            'cardamom': 'green cardamom pods',
            'mint': 'fresh mint leaves',
            'lemon': 'fresh lemon fruit',
            'garlic': 'fresh garlic cloves',
            'neem': 'fresh neem leaves',
            'aloe vera': 'fresh aloe vera leaf',
            'amla': 'fresh amla fruit',
            'ashwagandha': 'ashwagandha root natural',
            'cumin': 'cumin seeds natural',
            'fennel': 'fennel seeds natural'
        }
        
        # Get specific search term if available, otherwise use generic format
        search_query = query.lower()
        for key, value in search_terms.items():
            if key in search_query:
                search_query = value
                break
        else:
            search_query = f"natural fresh {query} ayurvedic ingredient -bottle -package -product"
        
        params = {
            "q": search_query,
            "cx": SEARCH_ENGINE_ID,
            "searchType": "image",
            "key": GOOGLE_API_KEY,
            "num": 1,
            "imgSize": "LARGE",
            "imgType": "photo",
            "safe": "active"
        }
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        if "items" in data:
            image_url = data["items"][0]["link"]
            image_cache[query] = image_url
            logger.debug(f"Successfully found image for {query}")
            return image_url
            
        logger.debug(f"No images found for {query}")
        return None
    except Exception as e:
        logger.error(f"Error fetching image for {query}: {str(e)}")
        return None

def generate_ayurvedic_image(query):
    """Generate Ayurvedic image using OpenAI's DALL-E."""
    try:
        response = openai.Image.create(
            prompt=f"High-quality image of {query} used in Ayurveda.",
            n=1,
            size="512x512"
        )
        return response["data"][0]["url"]
    except Exception as e:
        logger.error(f"Error generating image with DALL-E: {str(e)}")
        return None

def get_remedy_with_image(remedy_name):
    """Get images for a valid remedy with proper validation."""
    if not remedy_name or remedy_name.lower() not in known_ingredients:
        return []

    try:
        image_urls = []
        google_image = get_ayurvedic_image(remedy_name)
        if google_image:
            image_urls.append(google_image)
        
        if not image_urls:
            ai_image = generate_ayurvedic_image(remedy_name)
            if ai_image:
                image_urls.append(ai_image)
        
        return image_urls[:3]  # Limit to 3 images max
    except Exception as e:
        logger.error(f"Error in get_remedy_with_image for {remedy_name}: {str(e)}")
        return []

def clean_up_sentence(sentence):
    """Clean and lemmatize input sentence."""
    try:
        words = sentence.lower().split()
        return [lemmatizer.lemmatize(word) for word in words]
    except Exception as e:
        logger.error(f"Error in clean_up_sentence: {str(e)}")
        return []

def bow(sentence, words, show_details=False):
    """Convert sentence to bag of words."""
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        logger.debug(f"found in bag: {w}")
        return np.array(bag)
    except Exception as e:
        logger.error(f"Error in bow: {str(e)}")
        return np.zeros(len(words))

def predict_class(sentence, model):
    """Predict intent class for input sentence."""
    try:
        logger.debug(f"Predicting class for: {sentence}")
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
        logger.debug(f"Prediction results: {return_list}")
        return return_list
    except Exception as e:
        logger.error(f"Error in predict_class: {str(e)}")
        return []

def getResponse(ints, intents_json):
    """Get response based on predicted intent."""
    try:
        if not ints:
            logger.debug("No intents found")
            return "I'm not sure how to help with that. Could you please ask about a specific health concern?", ["unknown"]
        
        tag = ints[0]['intent']
        logger.debug(f"Getting response for intent: {tag}")
        list_of_intents = intents_json['intents']
        
        for i in list_of_intents:
            if i['tag'].lower() == tag.lower():  # Case-insensitive comparison
                response = random.choice(i['responses'])
                remedies = i.get('remedy', ["unknown"])
                
                # Format remedies list for better display
                if remedies != ["unknown"]:
                    remedy_list = ", ".join([r.title() for r in remedies[:-1]])
                    if len(remedies) > 1:
                        remedy_list += f" and {remedies[-1].title()}"
                    else:
                        remedy_list = remedies[0].title()
                    
                    # Add remedy names to response if not already mentioned
                    if not any(remedy.lower() in response.lower() for remedy in remedies):
                        response += f"\n\nKey ingredients: {remedy_list}"
                
                logger.debug(f"Found response: {response} with remedies: {remedies}")
                return response, remedies
        
        logger.debug(f"No response found for intent: {tag}")
        return "I'm not sure how to help with that specific concern. Could you please provide more details about your health issue?", ["unknown"]
    except Exception as e:
        logger.error(f"Error in getResponse: {str(e)}")
        return "I apologize, but I encountered an error. Please try rephrasing your health concern.", ["unknown"]

def get_llama_response(message):
    """Get response from LLaMA model with proper error handling."""
    try:
        # Check if it's a health-related query
        health_keywords = [
            'health', 'pain', 'ache', 'disease', 'remedy', 'cure', 'treatment',
            'medicine', 'ayurvedic', 'ayurveda', 'body', 'immunity', 'weight',
            'sleep', 'digestion', 'stress', 'anxiety', 'tired', 'fatigue',
            'energy', 'wellness', 'healing', 'natural', 'herbs', 'herbal',
            'throat', 'cough', 'cold', 'fever', 'headache', 'stomach'
        ]
        
        is_health_related = any(keyword in message.lower() for keyword in health_keywords)
        
        if not is_health_related:
            return "I am an Ayurvedic health assistant. I can only help you with health-related concerns. Please feel free to ask me about any health issues you're experiencing."

        # Simplified prompt for natural responses
        prompt = f"""You are an Ayurvedic expert. The user is experiencing {message}. 
Provide a simple Ayurvedic remedy in a natural, conversational way. Follow this example format:

"Take 1 teaspoon of fresh ginger juice and 1 teaspoon of lemon juice. Add a pinch of black salt and ½ teaspoon of zeera powder. Take this mixture 3 times after meals. This will help with indigestion."

Keep your response:
1. In simple, clear sentences
2. Include ingredients and their quantities
3. Include preparation method
4. Include dosage and timing
5. End with the benefit
6. Keep it brief (2-3 sentences)
7. Do not use labels like 'Remedy:', 'Ingredients:', etc.
8. Write in a natural, flowing way

Remember to keep the response very brief and easy to understand."""

        response = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {"role": "user", "content": message}
            ],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 512
            }
        )
        
        # Clean up the response to ensure it's in the right format
        response_text = response['message']['content'].strip()
        
        # Remove any remaining labels if they exist
        response_text = response_text.replace('Remedy:', '').replace('Ingredients:', '').replace('Method:', '').replace('Dosage:', '')
        
        # Ensure it starts with a capital letter and ends with a period
        response_text = response_text[0].upper() + response_text[1:]
        if not response_text.endswith('.'):
            response_text += '.'
            
        return response_text
    except Exception as e:
        logger.error(f"LLaMA Error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try asking about a specific health concern."

def translate_text(text, target_language):
    """Translate text to target language with proper error handling."""
    try:
        if not text or len(text) > 500:  # Limit text length for translation
            return text
            
        # Clean the text before translation
        text = text.replace('"', '').replace('"', '')  # Remove smart quotes
        text = text.replace('–', '-').replace('—', '-')  # Normalize dashes
        
        translator = Translator(to_lang=target_language)
        translated = translator.translate(text)
        
        # Clean the translated text
        translated = translated.replace('"', '').replace('"', '')
        translated = translated.replace('–', '-').replace('—', '-')
        
        return translated
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text

def extract_remedies_from_text(text):
    """Extract known remedies from text."""
    if not text:
        return ["unknown"]
    found = [
        ingredient for ingredient in known_ingredients 
        if f" {ingredient.lower()} " in f" {text.lower()} "
    ]
    return found if found else ["unknown"]

def chatbot_response(msg):
    """Generate chatbot response with proper logging."""
    try:
        logger.info(f"Processing message: {msg}")
        
        # Check if user is asking for alternative remedies
        if "alternative" in msg.lower() or "not satisfied" in msg.lower() or "another" in msg.lower():
            logger.info("User requesting alternative remedy, using LLaMA")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        # Check for multiple symptoms
        symptoms = ["fever", "headache", "cough", "cold", "stomach", "pain", "digestion"]
        found_symptoms = [s for s in symptoms if s in msg.lower()]
        
        if len(found_symptoms) > 1:
            logger.info(f"Multiple symptoms detected: {found_symptoms}, using LLaMA")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        # For single symptoms, try the model first
        ints = predict_class(msg, model)
        logger.info(f"Intent prediction: {ints}")

        if not ints or float(ints[0]['probability']) < 0.85:
            logger.info("Using LLaMA fallback")
            gpt_reply = get_llama_response(msg)
            extracted_remedies = extract_remedies_from_text(gpt_reply)
            return gpt_reply, extracted_remedies

        response, remedies = getResponse(ints, intents)
        return response, remedies
    except Exception as e:
        logger.error(f"Error in chatbot_response: {str(e)}")
        return "I apologize, but I'm having trouble understanding. Could you rephrase that?", ["unknown"]

@app.route('/update_chat_title/<int:session_id>', methods=['POST'])
@login_required
def update_chat_title(session_id):
    try:
        data = request.get_json()
        new_title = data.get('title')
        
        if not new_title:
            return jsonify({'success': False, 'error': 'No title provided'}), 400
            
        session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
        session.title = new_title
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating chat title: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    session_id = data.get('session_id')
    message_id = data.get('message_id')
    feedback = data.get('feedback')
    rating = data.get('rating')
    
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO feedback (session_id, message_id, feedback, rating, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (session_id, message_id, feedback, rating))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add feedback table to database
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            feedback TEXT,
            rating INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id),
            FOREIGN KEY (message_id) REFERENCES messages (id)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    with app.app_context():
        db.create_all()
    app.run(debug=True)
