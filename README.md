# AI Grievance Redressal Agent for Government Schemes

A professional Flask-based web application designed to help Indian citizens check eligibility and navigate government welfare schemes like **PM-Kisan**, **Ration Card**, and **Pension Schemes**.

## 🌟 Key Features

- **Multilingual Support**: Fully localized interface and AI responses in **English**, **Hindi**, and **Telugu**.
- **Smart Eligibility Checker**: Rule-based engine combined with AI to determine eligibility based on age, income, landholding, and more.
- **Voice Interactions**: 
  - **Voice Input**: Search for schemes and ask questions using your voice (Powered by Web Speech API).
  - **Text-to-Speech (TTS)**: The system reads eligibility results and guidance aloud in your native language.
- **AI Chatbot**: A persistent AI assistant to answer follow-up questions about application processes and documents.
- **Mobile Responsive**: Optimized for use on smartphones, tablets, and desktops.
- **Secure Configuration**: Environment-based configuration to protect sensitive API keys.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: Scikit-Learn (TF-IDF + Naive Bayes for routing), OpenRouter API (LLM integration)
- **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript
- **Translation**: Google Translator (Deep Translator)
- **Detection**: Langdetect

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- An [OpenRouter](https://openrouter.ai/) API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd battini
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
LLM_MODEL=openai/gpt-4o-mini
```

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

- `app.py`: Main Flask application and API routes.
- `agent.py`: Core AI logic for query processing and LLM integration.
- `matcher.py`: Rule-based eligibility checking engine.
- `static/`: CSS, images, and localized JavaScript (voice input, TTS).
- `templates/`: HTML templates for the user interface.
- `scheme_router_model.joblib`: Pre-trained model for classifying queries into scheme categories.

---
© 2023 Ministry of Rural Development, Government of India.
Developed for the AI Grievance Redressal Hackathon.