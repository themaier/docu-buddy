def create_analysis_prompt(context: str) -> str:
    """Create the prompt for LLM analysis"""

    prompt = f"""You are an expert code reviewer analyzing function complexity. Your goal is to provide DIFFERENTIATED ratings that distinguish between functions of varying complexity levels.

CONTEXT:
{context}

ANALYSIS REQUIREMENTS:

Rate each aspect on a 1-10 scale. BE SPECIFIC and use the FULL RANGE of scores:
- Use 1-3 for simple/excellent code
- Use 4-6 for moderate complexity  
- Use 7-8 for high complexity
- Use 9-10 for extremely complex/problematic code

**IMPORTANT: Functions should receive DIFFERENT scores based on their actual complexity. Avoid giving similar ratings to all functions.**

1. **Semantic Complexity** (1-10): How difficult is the logic to understand?
   - 1-3: Simple logic, clear algorithm, minimal domain knowledge needed
   - 4-6: Moderate logic with some complexity, reasonable algorithm
   - 7-8: Complex business logic, intricate algorithms, domain expertise needed
   - 9-10: Extremely complex logic, multiple interacting algorithms, expert-level domain knowledge

2. **Cognitive Load** (1-10): How much mental effort to comprehend?
   - 1-3: Easy to follow, minimal variable tracking, clear execution flow
   - 4-6: Some mental effort needed, moderate state tracking
   - 7-8: High mental effort, complex state management, difficult to trace execution
   - 9-10: Overwhelming mental effort, too many variables/states to track

3. **Maintainability** (1-10): How difficult would this be to modify safely?
   - 1-3: Easy to change, well-isolated, good testability
   - 4-6: Moderate change difficulty, some coupling
   - 7-8: Risky to change, high coupling, hard to test
   - 9-10: Extremely risky to modify, tightly coupled, change impact unpredictable

4. **Documentation Quality** (1-10): How well documented is this code?
   - 1-3: Excellent docs, clear comments, self-documenting
   - 4-6: Adequate documentation, some gaps
   - 7-8: Poor documentation, minimal comments
   - 9-10: No meaningful documentation, completely unclear

5. **Refactoring Urgency** (1-10): How urgently does this need refactoring?
   - 1-3: No refactoring needed, well-structured
   - 4-6: Minor improvements possible
   - 7-8: Should be refactored soon, causing some problems
   - 9-10: Critical refactoring needed immediately, major technical debt

6. **Function Descriptions**: Provide two different explanations of what this function does:
   - **Business Description**: Explain in simple, non-technical terms what business purpose this function serves. Focus on WHAT it accomplishes from a user/business perspective, avoiding technical jargon.
   - **Developer Description**: Provide a technical explanation of HOW the function works, including key algorithms, data structures, design patterns, and implementation details.

**CALIBRATION GUIDANCE:**
- Look at the structural metrics provided - they give you baseline complexity indicators
- A function with cyclomatic complexity of 15+ should likely get higher semantic scores
- Functions with 100+ lines should get higher cognitive load scores
- Functions with nesting depth 5+ should get higher maintainability concerns
- Compare this function's complexity to what you'd expect from typical enterprise code
- If function content is not available, base analysis primarily on the structural metrics provided

Please respond with ONLY the JSON (no other text):
{{
    "semantic_complexity": <number 1-10>,
    "cognitive_load": <number 1-10>,
    "maintainability": <number 1-10>,
    "documentation_quality": <number 1-10>,
    "refactoring_urgency": <number 1-10>,
    "explanation": "<2-3 sentence explanation of the main complexity drivers and why you gave these specific scores>",
    "business_description": "<Simple explanation of what this function does from a business perspective, avoiding technical terms>",
    "developer_description": "<Technical explanation of how the function works, including key implementation details and patterns>",
    "suggestions": [
        "<specific actionable suggestion 1>",
        "<specific actionable suggestion 2>",
        "<specific actionable suggestion 3>"
    ]
}}

Focus on what makes THIS SPECIFIC function more or less complex than average code."""

    return prompt
