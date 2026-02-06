"""
AI Interview Coach - Main Entry Point

A multi-agent system to help you prepare for technical interviews.
Uses RAG with your job descriptions and technical docs.

100% FREE - No paid API required:
  - LLM: Ollama (local models like Llama 3, Mistral, Gemma 2)
  - Embeddings: HuggingFace sentence-transformers (local)
  - Vector DB: ChromaDB (local)
  - Speech-to-text: Whisper (local, optional)
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from agents import InterviewerAgent, EvaluatorAgent, CoachAgent
from rag import VectorStore, DocumentLoader
from nlp import WHISPER_AVAILABLE

if WHISPER_AVAILABLE:
    from nlp import SpeechToText


class InterviewCoach:
    """
    Main orchestrator for the AI Interview Coach.
    
    Combines multiple agents:
    - Interviewer: Asks technical questions
    - Evaluator: Scores and critiques answers
    - Coach: Helps improve and learn
    
    Uses RAG to personalize based on job descriptions.
    100% free - runs entirely on local models (Ollama + HuggingFace).
    """
    
    # ---------- Function-calling tool definitions (LangChain-compatible) ----------
    TOOLS = [
        {
            "name": "start_interview",
            "description": "Start a new mock interview session on a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The interview topic"},
                },
                "required": ["topic"],
            },
        },
        {
            "name": "evaluate_answer",
            "description": "Evaluate the candidate's answer to the current question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The candidate answer"},
                },
                "required": ["answer"],
            },
        },
        {
            "name": "get_coaching",
            "description": "Get coaching advice on a topic or concern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or concern"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "generate_study_plan",
            "description": "Create a personalized study plan for interview preparation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Days until the interview"},
                },
                "required": ["days"],
            },
        },
    ]
    
    def __init__(self, model: str = "llama3.1"):
        """
        Initialize the Interview Coach.
        
        Args:
            model: Ollama model to use for agents (e.g. llama3, mistral, gemma2)
        """
        print("🚀 Initializing AI Interview Coach (100% Free - Local Models)...")
        
        # Verify Ollama is running
        self._check_ollama()
        
        # Initialize agents
        print("   🤖 Loading Interviewer Agent...")
        self.interviewer = InterviewerAgent(model=model)
        
        print("   📊 Loading Evaluator Agent...")
        self.evaluator = EvaluatorAgent(model=model)
        
        print("   🎓 Loading Coach Agent...")
        self.coach = CoachAgent(model=model)
        
        # Initialize RAG components
        print("   📚 Loading Knowledge Base (HuggingFace embeddings)...")
        self.vector_store = VectorStore()
        self.doc_loader = DocumentLoader()
        
        # Initialize Whisper (optional)
        self.stt = None
        if WHISPER_AVAILABLE:
            print("   🎙️ Whisper speech-to-text available")
        else:
            print("   ℹ️ Whisper not installed (optional: pip install openai-whisper)")
        
        # Session state
        self.job_description = ""
        self.chat_history = []
        self.current_question = ""
        self.scores = []
        self.model_name = model
        
        print(f"✅ AI Interview Coach ready! (model: {model})")
        print(f"   💰 Cost: $0.00 - everything runs locally\n")
    
    @staticmethod
    def _check_ollama():
        """Verify that Ollama is running locally."""
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            print("❌ Error: Ollama is not running!")
            print("   1. Install Ollama from: https://ollama.com")
            print("   2. Start it with: ollama serve")
            print("   3. Pull a model: ollama pull llama3")
            print("   Ollama is 100% free and runs models locally.")
            sys.exit(1)
    
    def dispatch_tool(self, tool_name: str, args: dict) -> str:
        """
        Function-calling dispatcher: routes tool calls to the right method.
        Demonstrates experience with function calls / tool-use patterns.
        
        Args:
            tool_name: Name of the tool to invoke
            args: Arguments for the tool
            
        Returns:
            Result string from the tool execution
        """
        handlers = {
            "start_interview": lambda a: self.start_interview(a["topic"]),
            "evaluate_answer": lambda a: json.dumps(self.answer_question(a["answer"])),
            "get_coaching": lambda a: self.coach.get_advice(a["query"]),
            "generate_study_plan": lambda a: self.get_study_plan(a.get("days", 7)),
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        return handler(args)
    
    def load_job_description(self, text: str = None, file_path: str = None) -> None:
        """
        Load a job description for personalized interview prep.
        
        Args:
            text: Raw job description text
            file_path: Path to job description file
        """
        if file_path:
            docs = self.doc_loader.load_file(file_path)
            self.job_description = "\n".join([d.page_content for d in docs])
            self.vector_store.add_documents(docs)
        elif text:
            self.job_description = text
            docs = self.doc_loader.load_text(text, metadata={"type": "job_description"})
            self.vector_store.add_documents(docs)
        
        print(f"✅ Job description loaded ({len(self.job_description)} chars)")
    
    def load_knowledge_base(self, directory: str) -> None:
        """
        Load technical documents for RAG-powered answers.
        
        Args:
            directory: Path to folder with technical docs
        """
        docs = self.doc_loader.load_directory(directory)
        if docs:
            self.vector_store.add_documents(docs)
            print(f"✅ Knowledge base loaded: {len(docs)} chunks")
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from the knowledge base."""
        results = self.vector_store.search(query, k=k)
        if not results:
            return "No additional context available."
        return "\n\n".join([doc.page_content for doc in results])
    
    def start_interview(self, topic: str = "general") -> str:
        """
        Start an interview session.
        
        Args:
            topic: The topic to focus on
            
        Returns:
            The first interview question
        """
        self.chat_history = []
        self.scores = []
        
        context = self.get_context(topic)
        
        question = self.interviewer.ask(
            topic=topic,
            job_context=self.job_description or "General AI/ML interview",
            tech_context=context,
            chat_history="Starting a new interview session."
        )
        
        self.current_question = question
        self.chat_history.append({"role": "interviewer", "content": question})
        
        return question
    
    def answer_question(self, answer: str) -> dict:
        """
        Submit an answer and get feedback.
        
        Args:
            answer: Your answer to the current question
            
        Returns:
            Dict with evaluation, score, and coaching tips
        """
        if not self.current_question:
            return {"error": "No active question. Use start_interview() first."}
        
        self.chat_history.append({"role": "candidate", "content": answer})
        
        # Get reference context for evaluation
        context = self.get_context(self.current_question)
        
        # Evaluate the answer
        evaluation = self.evaluator.evaluate(
            question=self.current_question,
            answer=answer,
            reference_context=context
        )
        
        score = self.evaluator.quick_score(self.current_question, answer)
        self.scores.append(score)
        
        # Get coaching tips
        coaching = self.coach.improve_answer(
            question=self.current_question,
            answer=answer,
            feedback=evaluation,
            reference=context
        )
        
        return {
            "question": self.current_question,
            "your_answer": answer,
            "score": score,
            "evaluation": evaluation,
            "coaching": coaching,
            "average_score": sum(self.scores) / len(self.scores) if self.scores else 0
        }
    
    def next_question(self, topic: str = None) -> str:
        """
        Get the next interview question.
        
        Args:
            topic: Optional new topic, or continue with current topic
            
        Returns:
            The next question
        """
        history = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}..."
            for msg in self.chat_history[-6:]  # Last 3 exchanges
        ])
        
        context = self.get_context(topic or "technical interview")
        
        question = self.interviewer.ask(
            topic=topic or "follow-up",
            job_context=self.job_description,
            tech_context=context,
            chat_history=history
        )
        
        self.current_question = question
        self.chat_history.append({"role": "interviewer", "content": question})
        
        return question
    
    def get_study_plan(self, days: int = 7) -> str:
        """
        Generate a personalized study plan.
        
        Args:
            days: Days until your interview
            
        Returns:
            A detailed study plan
        """
        return self.coach.create_study_plan(
            job_description=self.job_description or "General AI/ML position",
            days_until_interview=days
        )
    
    def explain(self, concept: str) -> str:
        """
        Get an explanation of a concept.
        
        Args:
            concept: The concept to explain
            
        Returns:
            Interview-ready explanation
        """
        return self.coach.explain_concept(concept)
    
    def generate_questions(self, num: int = 5) -> list[str]:
        """
        Generate practice questions based on the job description.
        
        Args:
            num: Number of questions
            
        Returns:
            List of interview questions
        """
        return self.interviewer.generate_questions(
            job_description=self.job_description or "AI/ML Engineer position",
            num_questions=num
        )


def interactive_mode():
    """Run an interactive interview session."""
    print("=" * 60)
    print("🎯 AI INTERVIEW COACH - Interactive Mode (Free / Local)")
    print("=" * 60)
    
    # Let user choose model
    print("\n🤖 Available Ollama models (make sure you've pulled one):")
    print("   - llama3.1  (default, 8B params)")
    print("   - mistral   (7B params, fast)")
    print("   - gemma2    (9B params, Google)")
    print("   - phi3      (3.8B params, lightweight)")
    print("\nModel name (press Enter for llama3.1): ", end="")
    model_choice = input().strip() or "llama3.1"
    
    coach = InterviewCoach(model=model_choice)
    
    # Optional: Load job description
    print("\n📋 Do you have a job description to load? (y/n): ", end="")
    if input().lower() == 'y':
        print("Enter path to file, or paste text (end with empty line):")
        first_line = input().strip()
        if Path(first_line).exists():
            coach.load_job_description(file_path=first_line)
        else:
            lines = [first_line]
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            if lines:
                coach.load_job_description(text="\n".join(lines))
    
    # Check for voice mode
    voice_mode = False
    if WHISPER_AVAILABLE:
        print("\n🎙️ Enable voice mode? Answer questions by speaking (y/n): ", end="")
        if input().lower() == 'y':
            coach.stt = SpeechToText(model_size="base")
            voice_mode = True
            print("✅ Voice mode enabled! You'll speak your answers.")
    
    print("\n" + "=" * 60)
    print("COMMANDS:")
    print("  'start [topic]' - Start interview on a topic")
    print("  'next'          - Get next question")
    print("  'explain X'     - Explain a concept")
    print("  'plan [days]'   - Get study plan")
    print("  'questions'     - Generate practice questions")
    if WHISPER_AVAILABLE:
        print("  'voice'         - Toggle voice mode on/off")
    print("  'tools'         - List available function-calling tools")
    print("  'quit'          - Exit")
    print("=" * 60)
    
    while True:
        print("\n>>> ", end="")
        user_input = input().strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\n👋 Good luck with your interview!")
            break
        
        elif user_input.lower() == 'tools':
            print("\n🔧 Available function-calling tools:")
            for tool in InterviewCoach.TOOLS:
                params = ", ".join(tool["parameters"]["properties"].keys())
                print(f"   • {tool['name']}({params}) - {tool['description']}")
        
        elif user_input.lower() == 'voice' and WHISPER_AVAILABLE:
            voice_mode = not voice_mode
            if voice_mode and coach.stt is None:
                coach.stt = SpeechToText(model_size="base")
            print(f"🎙️ Voice mode: {'ON' if voice_mode else 'OFF'}")
        
        elif user_input.lower().startswith('start'):
            topic = user_input[5:].strip() or "general AI/ML"
            print(f"\n🎤 Starting interview on: {topic}\n")
            question = coach.start_interview(topic)
            print(f"📝 INTERVIEWER:\n{question}\n")
            
            # Get answer (voice or text)
            if voice_mode and coach.stt:
                print("🎙️ Speak your answer (30 seconds max)...")
                answer = coach.stt.record_and_transcribe(duration=30)
                print(f"📝 You said: {answer}")
            else:
                print("Your answer: ", end="")
                answer = input()
            
            if answer:
                result = coach.answer_question(answer)
                print(f"\n⭐ Score: {result['score']}/10")
                print(f"\n📊 EVALUATION:\n{result['evaluation']}")
                print(f"\n🎓 COACHING:\n{result['coaching']}")
        
        elif user_input.lower() == 'next':
            question = coach.next_question()
            print(f"\n📝 INTERVIEWER:\n{question}\n")
            
            # Get answer (voice or text)
            if voice_mode and coach.stt:
                print("🎙️ Speak your answer (30 seconds max)...")
                answer = coach.stt.record_and_transcribe(duration=30)
                print(f"📝 You said: {answer}")
            else:
                print("Your answer: ", end="")
                answer = input()
            
            if answer:
                result = coach.answer_question(answer)
                print(f"\n⭐ Score: {result['score']}/10 (Average: {result['average_score']:.1f})")
                print(f"\n📊 EVALUATION:\n{result['evaluation']}")
        
        elif user_input.lower().startswith('explain'):
            concept = user_input[7:].strip()
            if concept:
                print(f"\n📚 Explaining: {concept}\n")
                explanation = coach.explain(concept)
                print(explanation)
            else:
                print("Usage: explain <concept>")
        
        elif user_input.lower().startswith('plan'):
            days = user_input[4:].strip()
            days = int(days) if days.isdigit() else 7
            print(f"\n📅 Creating {days}-day study plan...\n")
            plan = coach.get_study_plan(days)
            print(plan)
        
        elif user_input.lower() == 'questions':
            print("\n🎯 Generating practice questions...\n")
            questions = coach.generate_questions(5)
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")
        
        else:
            # Treat as a question/comment and get coaching advice
            advice = coach.coach.get_advice(user_input)
            print(f"\n🎓 COACH:\n{advice}")


if __name__ == "__main__":
    interactive_mode()
