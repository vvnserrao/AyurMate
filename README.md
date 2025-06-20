# AyurMate

AyurMate is a modern, AI-powered Ayurvedic chatbot web application that blends traditional Ayurvedic wisdom with advanced technology. It offers multilingual support, user authentication, persistent chat history, and a visually appealing, professional user interface.

## 🌿 Project Overview

AyurMate is designed to make Ayurvedic knowledge accessible to everyone. Users can interact with the chatbot in English, Hindi, or Kannada, receive personalized remedies (with images), and manage their chat history securely. The application is built with Flask, SQLAlchemy, and a neural network for intent detection.

## ✨ Features

- **Multilingual Support:** Communicate in English, Hindi, or Kannada. The chatbot translates queries and responses for seamless interaction.
- **Voice Interaction:** Speak to the bot and receive spoken responses (including Hindi and Kannada).
- **Visual Remedies:** Suggested remedies include images and names for better understanding.
- **User Authentication:** Secure login, registration, and logout with session management.
- **Persistent Chat History:** Each user's chat sessions are saved, can be reviewed, managed (delete/download sessions).
- **Modern UI:** Responsive, attractive design consistent across landing, login, and chat pages.
- **Session Management:** Sidebar for switching between chat sessions, with remedy images and names displayed per message.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AyurBot
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data (if not already present):**
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   ```
5. **Set up environment variables:**
   - Create a `.env` file in the root directory and add your API keys and secret key:
     ```env
     SECRET_KEY=your-secret-key
     GOOGLE_API_KEY=your-google-api-key
     SEARCH_ENGINE_ID=your-search-engine-id
     OPENAI_API_KEY=your-openai-api-key
     ```
6. **Run the application:**
   ```bash
   python app.py
   ```

## 🖥️ Usage
- Visit the landing page to learn about AyurMate.
- Register or log in to access the chatbot.
- Start a new chat or review previous sessions in the sidebar.
- Download or delete chat sessions as needed.

## 🛠️ Tech Stack
- **Backend:** Flask, Flask-Login, Flask-SQLAlchemy, Keras, NLTK, OpenAI API
- **Frontend:** HTML5, CSS3, JavaScript
- **Database:** SQLite
- **Other:** Google Custom Search API, Deep Translator

## 📁 Folder Structure
```
AyurBot/
├── app.py                # Main Flask application
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, images)
├── templates/            # HTML templates
├── training.py           # Model training script
├── model.h5              # Trained neural network model (ignored in git)
├── data.json             # Intents and remedies data (ignored in git)
├── labels.pkl, texts.pkl # Model data (ignored in git)
├── nltk_data/            # NLTK data (ignored in git)
├── instance/             # Runtime files (ignored in git)
├── venv/                 # Virtual environment (ignored in git)
└── ...
```

## 📸 Screenshots

<img width="1727" alt="Home Page" src="https://github.com/user-attachments/assets/710283ae-bd69-4fab-a10c-3e9eb6c4acb3" />

**Landing Page*****
<img width="1727" alt="data set response" src="https://github.com/user-attachments/assets/67cbf43f-8eab-406f-98ca-8de90dbe8a5d" />

**Chat Interface**


| Landing Page | Chat Interface | Remedy Example | Login/Register |
|:------------:|:--------------:|:--------------:|:--------------:|
| ![](screenshots/landing.png) | ![](screenshots/chat.png) | ![](screenshots/remedy.png) | ![](screenshots/login.png) |

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- [Vivian Serrao](https://www.vivianserrao.com/) for development
- [NLTK](https://www.nltk.org/), [Flask](https://flask.palletsprojects.com/), [Keras](https://keras.io/), [OpenAI](https://openai.com/), [Deep Translator](https://pypi.org/project/deep-translator/)

---
© 2025 AyurMate - Developed by Vivian Serrao

