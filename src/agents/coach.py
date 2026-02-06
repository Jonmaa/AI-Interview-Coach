"""
Coach Agent - Provides guidance and helps improve answers.
Acts as a mentor during interview preparation.
Uses Ollama (free, local LLM) - no paid API required.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


COACH_PROMPT = """You are an expert interview coach helping candidates prepare for technical interviews.
You are supportive, encouraging, and provide actionable advice.

CANDIDATE'S QUESTION OR CONCERN:
{user_input}

CONTEXT (if available):
{context}

JOB REQUIREMENTS:
{job_requirements}

As a coach, you should:
1. Acknowledge the candidate's effort
2. Provide specific, actionable advice
3. Share frameworks or techniques for answering
4. Give examples when helpful
5. Be encouraging but honest
6. Help them understand what interviewers are looking for

Respond in a warm, mentor-like tone."""


IMPROVE_ANSWER_PROMPT = """You are an expert interview coach helping candidates improve their answers.

ORIGINAL QUESTION:
{question}

CANDIDATE'S ANSWER:
{answer}

EVALUATION FEEDBACK:
{feedback}

REFERENCE MATERIAL:
{reference}

Your task:
1. Acknowledge what was good about the answer
2. Identify the key missing elements
3. Provide a IMPROVED VERSION of the answer
4. Explain WHY the improved version is better
5. Give tips for remembering key points

Format your response as:

## What You Did Well
[positive points]

## Key Improvements Needed
[specific improvements]

## Improved Answer
[A complete, improved version of the answer]

## Why This Is Better
[Explanation of improvements]

## Pro Tips
[Memorable tips for this topic]"""


class CoachAgent:
    """
    The Coach Agent helps candidates improve their interview skills.
    
    Usage:
        coach = CoachAgent()
        advice = coach.get_advice("How do I explain RAG?")
        improved = coach.improve_answer(question, answer, feedback)
    """
    
    def __init__(self, model: str = "llama3.1"):
        """
        Initialize the coach agent.
        
        Args:
            model: Ollama model to use (e.g. llama3.1, mistral, gemma2)
        """
        self.llm = ChatOllama(model=model, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_template(COACH_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def get_advice(
        self,
        user_input: str,
        context: str = "",
        job_requirements: str = "General AI/ML position"
    ) -> str:
        """
        Get coaching advice.
        
        Args:
            user_input: The candidate's question or concern
            context: Additional context
            job_requirements: The job they're preparing for
            
        Returns:
            Coaching advice
        """
        return self.chain.invoke({
            "user_input": user_input,
            "context": context,
            "job_requirements": job_requirements
        })
    
    def improve_answer(
        self,
        question: str,
        answer: str,
        feedback: str = "",
        reference: str = ""
    ) -> str:
        """
        Help improve an interview answer.
        
        Args:
            question: The interview question
            answer: The candidate's original answer
            feedback: Evaluation feedback (optional)
            reference: Reference material for good answers
            
        Returns:
            Improved answer with explanation
        """
        prompt = ChatPromptTemplate.from_template(IMPROVE_ANSWER_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "question": question,
            "answer": answer,
            "feedback": feedback or "No specific feedback provided.",
            "reference": reference or "Use your expertise."
        })
    
    def create_study_plan(
        self,
        job_description: str,
        experience_level: str = "mid",
        days_until_interview: int = 7
    ) -> str:
        """
        Create a personalized study plan.
        
        Args:
            job_description: The job requirements
            experience_level: junior, mid, or senior
            days_until_interview: Time available to prepare
            
        Returns:
            A detailed study plan
        """
        prompt = ChatPromptTemplate.from_template("""
Create a {days}-day interview preparation plan.

JOB DESCRIPTION:
{job_description}

CANDIDATE LEVEL: {level}

Create a day-by-day study plan that:
1. Prioritizes the most important topics
2. Balances theory and practice
3. Includes specific resources to study
4. Has daily practice exercises
5. Builds confidence progressively

Format as a clear, actionable daily schedule.""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "job_description": job_description,
            "level": experience_level,
            "days": days_until_interview
        })
    
    def explain_concept(self, concept: str, depth: str = "interview") -> str:
        """
        Explain a technical concept at interview depth.
        
        Args:
            concept: The concept to explain
            depth: brief, interview, or deep
            
        Returns:
            Explanation suitable for interviews
        """
        depth_instructions = {
            "brief": "Give a 2-3 sentence overview suitable for a quick mention.",
            "interview": "Explain as you would in an interview: clear, comprehensive, with examples.",
            "deep": "Provide a thorough technical explanation with implementation details."
        }
        
        prompt = ChatPromptTemplate.from_template("""
Explain this concept for an AI/ML interview:

CONCEPT: {concept}

DEPTH: {instruction}

Include:
1. Clear definition
2. Why it matters
3. A concrete example
4. Common follow-up questions interviewers might ask""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "concept": concept,
            "instruction": depth_instructions.get(depth, depth_instructions["interview"])
        })


if __name__ == "__main__":
    print("=" * 50)
    print("COACH AGENT TEST")
    print("=" * 50)
    
    coach = CoachAgent()
    
    # Test concept explanation
    print("\n📚 Explaining 'Function Calling'...")
    explanation = coach.explain_concept("Function Calling in LLMs", depth="interview")
    print(explanation)
    
    print("\n" + "=" * 50)
    
    # Test answer improvement
    print("\n🔄 Improving an answer...")
    
    question = "What is LangChain?"
    bad_answer = "LangChain is a library for AI."
    
    improved = coach.improve_answer(
        question=question,
        answer=bad_answer,
        feedback="Too brief, lacks specifics"
    )
    print(improved)
    
    print("\n✅ Test completed!")
