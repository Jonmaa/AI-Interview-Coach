"""
Interviewer Agent - Conducts technical interviews.
Asks relevant questions based on job descriptions and tech knowledge.
Uses Ollama (free, local LLM) - no paid API required.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


INTERVIEWER_PROMPT = """You are an expert technical interviewer for AI/ML positions.
Your role is to ask challenging but fair technical questions.

CONTEXT FROM JOB DESCRIPTION:
{job_context}

RELEVANT TECHNICAL KNOWLEDGE:
{tech_context}

INTERVIEW GUIDELINES:
1. Start with a warm greeting and explain the interview format
2. Ask questions that test both theoretical knowledge and practical experience
3. Follow up on answers to dig deeper
4. Cover topics relevant to the job description
5. Mix conceptual questions with practical scenarios
6. Be professional and encouraging

PREVIOUS CONVERSATION:
{chat_history}

CURRENT TOPIC: {topic}

Generate the next interview question or follow-up. Be specific and technical.
If starting fresh, introduce yourself first."""


class InterviewerAgent:
    """
    The Interviewer Agent conducts technical interviews.
    
    Usage:
        interviewer = InterviewerAgent()
        question = interviewer.ask(
            topic="LangChain agents",
            job_context="...",
            tech_context="...",
            chat_history="..."
        )
    """
    
    def __init__(self, model: str = "llama3.1"):
        """
        Initialize the interviewer agent.
        
        Args:
            model: Ollama model to use (e.g. llama3.1, mistral, gemma2)
        """
        self.llm = ChatOllama(model=model, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_template(INTERVIEWER_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def ask(
        self,
        topic: str = "general AI/ML",
        job_context: str = "No specific job description provided.",
        tech_context: str = "No additional technical context.",
        chat_history: str = "This is the start of the interview."
    ) -> str:
        """
        Generate an interview question.
        
        Args:
            topic: The current interview topic
            job_context: Relevant context from job description
            tech_context: Technical knowledge from RAG
            chat_history: Previous conversation
            
        Returns:
            The next interview question or follow-up
        """
        response = self.chain.invoke({
            "topic": topic,
            "job_context": job_context,
            "tech_context": tech_context,
            "chat_history": chat_history
        })
        return response
    
    def generate_questions(
        self,
        job_description: str,
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> list[str]:
        """
        Generate a list of interview questions based on a job description.
        
        Args:
            job_description: The job requirements
            num_questions: Number of questions to generate
            difficulty: easy, medium, or hard
            
        Returns:
            List of interview questions
        """
        prompt = ChatPromptTemplate.from_template("""
Based on this job description, generate {num_questions} technical interview questions.
Difficulty level: {difficulty}

JOB DESCRIPTION:
{job_description}

Generate questions that:
1. Test practical experience with the required technologies
2. Assess problem-solving abilities
3. Evaluate understanding of core concepts
4. Include scenario-based questions

Format: Return ONLY the questions, numbered 1 to {num_questions}.
""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "job_description": job_description,
            "num_questions": num_questions,
            "difficulty": difficulty
        })
        
        # Parse questions from response
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove the number prefix
                question = line.split(".", 1)[-1].strip()
                if question:
                    questions.append(question)
        
        return questions[:num_questions]


if __name__ == "__main__":
    print("=" * 50)
    print("INTERVIEWER AGENT TEST")
    print("=" * 50)
    
    interviewer = InterviewerAgent()
    
    # Test question generation
    job_desc = """
    We are looking for an AI Engineer with:
    - 1+ years experience with Agentic AI development
    - Experience with LangChain and function calling
    - Knowledge of LLMs and embeddings
    - Python programming skills
    - Experience with RAG systems
    - NLP experience including Whisper
    """
    
    print("\n📝 Generating interview questions...")
    questions = interviewer.generate_questions(job_desc, num_questions=3)
    
    print("\n🎯 Generated questions:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
    
    print("\n✅ Test completed!")
