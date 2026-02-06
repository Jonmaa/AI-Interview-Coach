# 🎯 AI Interview Coach

An AI-powered interview preparation system using **multi-agent architecture**, **RAG**, and **local LLMs via Ollama** — **100% free, no paid APIs**.



## 🧠 What does this project do?

This system helps you prepare for technical interviews with:
1. **Personalized Questions** — Generates questions from YOUR job description
2. **Real-time Evaluation** — Scores and critiques your answers
3. **Expert Coaching** — Helps you improve with actionable feedback
4. **Knowledge Base** — Uses RAG to reference technical documentation
5. **Voice Mode** — Answer questions by speaking (Whisper STT)
6. **Function Calling** — Tool-use patterns for agentic dispatch

## 🏗️ Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI INTERVIEW COACH                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 🎤 Interviewer│  │ 📊 Evaluator │  │ 🎓 Coach     │       │
│  │   Agent      │  │    Agent     │  │    Agent     │       │
│  │              │  │              │  │              │       │
│  │ • Asks       │  │ • Scores     │  │ • Improves   │       │
│  │   questions  │  │   answers    │  │   answers    │       │
│  │ • Follows up │  │ • Feedback   │  │ • Explains   │       │
│  │ • Adapts     │  │ • Compares   │  │ • Plans      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └────────────┬────┴────────────────┘                │
│                      │                                       │
│              ┌───────┴───────┐                              │
│              │  📚 RAG System │                              │
│              │               │                              │
│              │ • Embeddings  │                              │
│              │ • ChromaDB    │                              │
│              │ • Documents   │                              │
│              └───────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technologies

| Component | Technology | Purpose | Cost |
|-----------|------------|---------|------|
| LLM | Ollama (Llama 3 / Mistral) | Multi-agent reasoning | Free |
| Embeddings | HuggingFace sentence-transformers | Semantic search | Free |
| Vector Store | ChromaDB | Document storage | Free |
| Framework | LangChain + LangGraph | Agent orchestration | Free |
| NLP / STT | OpenAI Whisper (local) | Voice input | Free |
| UI (optional) | Streamlit | Web interface | Free |

## 🤖 The Three Agents

### 🎤 Interviewer Agent
- Generates technical interview questions
- Adapts based on job description
- Asks follow-up questions
- Simulates real interview scenarios

### 📊 Evaluator Agent
- Scores answers (1-10)
- Identifies strengths and weaknesses
- Compares against reference knowledge
- Provides detailed feedback

### 🎓 Coach Agent
- Helps improve weak answers
- Explains concepts clearly
- Creates personalized study plans
- Mentors throughout preparation

## 📚 RAG System

The system uses **Retrieval Augmented Generation** with 100% local components:
1. **Load Documents** — Upload job descriptions, tech docs
2. **Create Embeddings** — Convert to vectors with **HuggingFace** (local, free)
3. **Store in ChromaDB** — Persistent vector database (local, free)
4. **Semantic Search** — Find relevant context for questions
5. **Augment Responses** — Use context to improve accuracy

## 🎙️ Voice Mode (Whisper NLP)

Answer interview questions by speaking! Uses **OpenAI Whisper** (the open-source model, NOT the paid API):
- Runs entirely locally — no data sent to any server
- Supports multiple languages
- Models: tiny (fastest) → large (most accurate)
- Toggle with the `voice` command during sessions

## 🔧 Function Calling / Tool Use

The system implements a **function-calling dispatcher** pattern:
```python
coach.dispatch_tool("start_interview", {"topic": "LangChain agents"})
coach.dispatch_tool("evaluate_answer", {"answer": "RAG combines..."})
coach.dispatch_tool("get_coaching", {"query": "How to explain RAG?"})
coach.dispatch_tool("generate_study_plan", {"days": 7})
```

## 📋 Requirements

- Python 3.10+
- **Ollama** installed and running (free: https://ollama.com)
- 2-4GB disk space (for models)
- No API keys needed!

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-interview-coach.git
cd ai-interview-coach
```

### 2. Install Ollama (free, local LLM runtime)
```bash
# Download from https://ollama.com then:
ollama pull llama3       # recommended (8B params)
# or alternatives:
# ollama pull mistral    # 7B, fast
# ollama pull gemma2     # 9B, Google
# ollama pull phi3       # 3.8B, lightweight
```

### 3. Create virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
# Make sure Ollama is running first:
ollama serve

# Then in another terminal:
python src/main.py
```

## 💻 Usage

### Interactive Mode
```bash
python src/main.py
```

Commands:
- `start [topic]` — Start interview on a topic
- `next` — Get next question
- `explain X` — Explain a concept
- `plan [days]` — Get study plan
- `questions` — Generate practice questions
- `voice` — Toggle voice mode (Whisper)
- `tools` — List function-calling tools
- `quit` — Exit

### Example Session
```
>>> start LangChain agents

📝 INTERVIEWER:
Hello! I'm excited to discuss LangChain agents with you today.
Let's start: Can you explain what a ReAct agent is and how it differs 
from a simple chain?

Your answer: A ReAct agent combines reasoning and acting...

⭐ Score: 7/10

📊 EVALUATION:
STRENGTHS:
- Good understanding of the core concept
- Mentioned the reasoning-acting loop

AREAS FOR IMPROVEMENT:
- Could include a concrete example
- Didn't mention tool integration
```

## 🧪 Test Individual Components

```bash
# Test embeddings (free, local HuggingFace)
python src/rag/embeddings.py

# Test vector store
python src/rag/vector_store.py

# Test document loader
python src/rag/document_loader.py

# Test interviewer agent (requires Ollama running)
python src/agents/interviewer.py

# Test evaluator agent (requires Ollama running)
python src/agents/evaluator.py

# Test coach agent (requires Ollama running)
python src/agents/coach.py

# Test Whisper speech-to-text
python src/nlp/whisper_stt.py
```

## 📁 Project Structure

```
ai-interview-coach/
├── src/
│   ├── main.py              # Main entry point & orchestrator
│   ├── agents/
│   │   ├── interviewer.py   # Interview question agent (Ollama)
│   │   ├── evaluator.py     # Answer evaluation agent (Ollama)
│   │   └── coach.py         # Coaching and improvement agent (Ollama)
│   ├── rag/
│   │   ├── embeddings.py    # HuggingFace embeddings (free, local)
│   │   ├── vector_store.py  # ChromaDB vector store
│   │   └── document_loader.py # Document processing
│   └── nlp/
│       ├── __init__.py      # NLP module exports
│       └── whisper_stt.py   # Whisper speech-to-text (free, local)
├── knowledge/
│   ├── job_descriptions/    # Your job descriptions
│   └── tech_docs/           # Technical reference docs
├── data/
│   └── chroma_db/           # Persistent vector database
├── requirements.txt
└── README.md
```

## 🎯 Skills Demonstrated

This project demonstrates proficiency in:

- ✅ **Agentic AI Development** — Multi-agent orchestration (Interviewer + Evaluator + Coach)
- ✅ **LangChain** — Chains, prompts, agents, output parsers
- ✅ **Generative AI / LLMs** — Local models via Ollama (Llama 3, Mistral, etc.)
- ✅ **Function Calling** — Tool-use dispatcher pattern for agentic workflows
- ✅ **RAG Systems** — HuggingFace embeddings + ChromaDB vector search
- ✅ **NLP / Whisper** — Speech-to-text for voice-based interview practice
- ✅ **Python** — Modern Python with type hints, clean architecture
- ✅ **System Design** — Modular, scalable, zero-cost architecture
- ✅ **Problem Solving** — Full-stack AI application without any paid APIs

## Why this kind of project?
I wanted to improve my knowledge in this area as I am having an interview related to this soon.