# GenAI Stack

A modern AI application stack using Gemini, Neo4j, and LangChain.

## Features

- 🤖 AI-powered chatbot using Google's Gemini Pro
- 📚 RAG (Retrieval Augmented Generation) with Neo4j vector store
- 📄 PDF document processing and Q&A
- 🌐 REST API for integration
- 💻 Modern web interface

## Prerequisites

- Docker and Docker Compose
- Google Cloud API key with Gemini API enabled
- Neo4j database (included in Docker setup)

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd genai-stack
```

2. Create a `.env` file with your configuration:
```env
LLM=gemini-pro
EMBEDDING_MODEL=google-genai-embedding-001
GOOGLE_API_KEY=your-google-api-key-here
```

3. Start the stack:
```bash
docker-compose up
```

## Services

- Frontend: http://localhost:8505
- Bot UI: http://localhost:8501
- PDF Bot: http://localhost:8503
- API: http://localhost:8504
- Neo4j Browser: http://localhost:7474

## Development

The project uses Docker Compose for development with hot-reloading enabled. Each service can be developed independently:

- `front-end/`: React-based web interface
- `bot.py`: Main chatbot interface
- `pdf_bot.py`: PDF processing and Q&A
- `api.py`: REST API endpoints
- `loader.py`: Data loading utilities

## License

Private - All rights reserved
