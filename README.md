# LangGraph Chatbot with Tools & Memory

This project is an **AI-powered chatbot** built using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/), and [Streamlit](https://streamlit.io/).  
It integrates multiple tools such as **web search, weather lookup, stock prices, Wikipedia summaries, and a calculator**, and supports **persistent memory across chat sessions** using SQLite.

---

##  Features

- ✅ **Conversational AI** powered by **Google Gemini** (`gemini-2.5-flash`).
- 🔧 **Integrated tools**:
  - 🌐 DuckDuckGo Web Search  
  - ☁️ Weather lookup (via OpenWeather API)  
  - 📈 Stock price fetcher (via Alpha Vantage API)  
  - 📚 Wikipedia search & summarizer  
  - ➕➖✖️➗ Calculator
- 💾 **Persistent chat memory** using `SqliteSaver` (sessions stored in `chatbot.db`).
- 🧵 **Multiple conversations/threads** with ability to rename threads automatically.
- 🎛️ **Interactive UI** using Streamlit with sidebar conversation management.
- ⚡ **Streaming responses** with live tool execution status updates.

---

##  Project Structure
- ├── backend_db.py # LangGraph + tools + chatbot definition
- ├── frontend_db.py # Streamlit frontend
- ├── chatbot.db 
- ├── .env # Environment variables (API keys)
- ├── requirements.txt # Dependencies
- └── README.md # Project documentation


---

##  Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/langgraph-chatbot.git
cd langgraph-chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
```bash
GOOGLE_API_KEY=your_google_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

## Run the App
``` bash
streamlit run app.py
```

## Usage

- Start a new chat via sidebar → "New Chat".
- Conversations are saved in chatbot.db and can be resumed anytime.
- The chatbot can automatically call tools when required (e.g., answering math problems, looking up stock prices, or fetching weather).
- Thread titles are auto-generated from the first user message.

## License

