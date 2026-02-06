"""
Evaluator Agent - Evaluates interview responses.
Provides scores and detailed feedback on answers.
Uses Ollama (free, local LLM) - no paid API required.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Structured evaluation result."""
    score: int  # 1-10
    strengths: list[str]
    weaknesses: list[str]
    technical_accuracy: str
    communication: str
    overall_feedback: str


EVALUATOR_PROMPT = """You are an expert technical interviewer evaluating candidate responses.
Provide honest, constructive feedback.

INTERVIEW QUESTION:
{question}

CANDIDATE'S ANSWER:
{answer}

REFERENCE KNOWLEDGE (what a good answer should include):
{reference_context}

EVALUATION CRITERIA:
1. Technical Accuracy (0-10): Is the answer technically correct?
2. Completeness (0-10): Does it cover all important aspects?
3. Clarity (0-10): Is the explanation clear and well-structured?
4. Practical Experience (0-10): Does it show real hands-on experience?
5. Communication (0-10): Is it well-articulated?

Provide your evaluation in this EXACT format:

SCORE: [overall score 1-10]

STRENGTHS:
- [strength 1]
- [strength 2]

AREAS FOR IMPROVEMENT:
- [weakness 1]
- [weakness 2]

TECHNICAL ACCURACY:
[Brief assessment of technical correctness]

COMMUNICATION:
[Brief assessment of how well they communicated]

WHAT A GREAT ANSWER WOULD INCLUDE:
[Key points that should be mentioned]

OVERALL FEEDBACK:
[Constructive summary with encouragement]"""


class EvaluatorAgent:
    """
    The Evaluator Agent assesses interview responses.
    
    Usage:
        evaluator = EvaluatorAgent()
        feedback = evaluator.evaluate(
            question="What is RAG?",
            answer="RAG stands for...",
            reference_context="..."
        )
    """
    
    def __init__(self, model: str = "llama3.1"):
        """
        Initialize the evaluator agent.
        
        Args:
            model: Ollama model to use (e.g. llama3.1, mistral, gemma2)
        """
        self.llm = ChatOllama(model=model, temperature=0.3)  # Lower temp for consistent evaluation
        self.prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def evaluate(
        self,
        question: str,
        answer: str,
        reference_context: str = "Use your knowledge to evaluate."
    ) -> str:
        """
        Evaluate an interview answer.
        
        Args:
            question: The interview question
            answer: The candidate's answer
            reference_context: Technical context from RAG for comparison
            
        Returns:
            Detailed evaluation feedback
        """
        response = self.chain.invoke({
            "question": question,
            "answer": answer,
            "reference_context": reference_context
        })
        return response
    
    def quick_score(self, question: str, answer: str) -> int:
        """
        Get a quick 1-10 score for an answer.
        
        Args:
            question: The interview question
            answer: The candidate's answer
            
        Returns:
            Score from 1-10
        """
        prompt = ChatPromptTemplate.from_template("""
Rate this interview answer from 1-10.
Question: {question}
Answer: {answer}

Respond with ONLY a number from 1 to 10.""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "answer": answer
        })
        
        # Parse the score
        try:
            score = int(response.strip().split()[0])
            return max(1, min(10, score))  # Clamp to 1-10
        except:
            return 5  # Default middle score
    
    def compare_answers(
        self,
        question: str,
        answer1: str,
        answer2: str
    ) -> str:
        """
        Compare two answers and explain which is better.
        
        Args:
            question: The interview question
            answer1: First answer
            answer2: Second answer (usually the improved version)
            
        Returns:
            Comparison analysis
        """
        prompt = ChatPromptTemplate.from_template("""
Compare these two answers to the same interview question.

QUESTION: {question}

ANSWER 1 (Original):
{answer1}

ANSWER 2 (Revised):
{answer2}

Analyze:
1. Which answer is better and why?
2. What improvements were made?
3. What could still be improved?

Be specific and constructive.""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "question": question,
            "answer1": answer1,
            "answer2": answer2
        })


if __name__ == "__main__":
    print("=" * 50)
    print("EVALUATOR AGENT TEST")
    print("=" * 50)
    
    evaluator = EvaluatorAgent()
    
    question = "What is RAG (Retrieval Augmented Generation) and why is it useful?"
    
    # Test with a mediocre answer
    mediocre_answer = "RAG is when you search for stuff and then give it to the AI."
    
    print(f"\n📝 Question: {question}")
    print(f"\n🎤 Answer: {mediocre_answer}")
    print("\n" + "=" * 50)
    print("EVALUATION:")
    print("=" * 50)
    
    feedback = evaluator.evaluate(question, mediocre_answer)
    print(feedback)
    
    # Quick score
    score = evaluator.quick_score(question, mediocre_answer)
    print(f"\n⭐ Quick Score: {score}/10")
    
    print("\n✅ Test completed!")
