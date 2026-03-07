def generate_report(llm, context, query):

    prompt = f"""
You are an AI research assistant.

Using the provided context generate a structured research report.

Topic: {query}

Context:
{context}

Report Structure:
1. Introduction
2. Key Concepts
3. Applications
4. Challenges
5. Future Scope
6. Conclusion
"""

    response = llm.invoke(prompt)

    return response.content