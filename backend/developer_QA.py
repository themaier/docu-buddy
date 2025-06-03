from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, AIMessagePromptTemplate

load_dotenv()

## llm agent ---------------------------------------------------------------------------
llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,
        max_tokens=None,
        timeout=None,
        max_retries=2
        # base_url="...",
        # organization="...",
        # other params...
    )


def get_developer_qa(code_dict: dict) -> str:
    
    system_prompt = """
    You are a smart, detail-oriented code analysis and explanation assistant. Your job is to analyze any code the user provides from a professional developer's perspective.

    You will be given two types of inputs:
    1. A block of code (code input) — your job is to analyze it thoroughly, understand its architecture, structure, modules, and logic, just like an experienced developer would during a code review.
    2. A query — a natural language question from the user asking about a particular aspect, module, function, or behavior of the previously submitted code.

    Your tasks are:
    - When the code is provided, perform a deep analysis and retain an internal structured understanding of the code: identify key modules, functions, classes, dependencies, logic flow, and potential improvements.
    - When a query is provided, respond only based on the code given, offering clear, technically sound, and accurate explanations related to the query.

    """
    
    query = """
    Based on the following inputs, analyze the provided code and answer the user’s query with a detailed technical explanation.

    Inputs:
    - Code Text: {code_text}
    - User Query: {user_query}
    """
    
    examples = [
        {
            "input": """
    Based on the following inputs, analyze the provided code and answer the user’s query with a detailed technical explanation.

    Inputs:
    - Code Text:
    def add_numbers(a, b):
        return a + b

    - User Query:
    What does the add_numbers function do?
    """,
            "output": """

    **Relevant Code**
    ```def add_numbers(a, b):
        return a + b```


            The add_numbers function is a fundamental utility that performs addition between two inputs, a and b. Key aspects include: - Purpose: Accepts two arguments and returns their sum using the '+' operator. - Input Expectations: Assumes that inputs are compatible types (integers, floats, or strings). - Design Simplicity: Minimalist structure without any error handling or type checking. - Reusability: Can be embedded in broader applications requiring arithmetic operations. - Limitations: Potential runtime errors if incompatible types (e.g., integer and string) are passed without pre-validation. In summary, this function is a clean, single-responsibility component ideal for controlled environments where input validation is managed externally.
            """


        },
        {
            "input": """
    Based on the following inputs, analyze the provided code and answer the user’s query with a detailed technical explanation.

    Inputs:
    - Code Text:
    class Calculator:
        def __init__(self):
            self.history = []

        def multiply(self, x, y):
            result = x * y
            self.history.append(('multiply', x, y, result))
            return result

    - User Query:
    Explain the multiply method and how history is used.
    """,
            "output": """

    **Relevant Code**
    ```class Calculator:
        def __init__(self):
            self.history = []

        def multiply(self, x, y):
            result = x * y
            self.history.append(('multiply', x, y, result))
            return result```

    The multiply method in the Calculator class performs a multiplication operation and logs each result. Key components are: - Functionality: Multiplies two inputs, x and y, and returns the product. - Operation Logging: Appends a tuple ('multiply', x, y, result) to the history list after each execution. - Data Structure: The history list maintains chronological records, facilitating auditability and reproducibility. - Design Pattern: Demonstrates encapsulation by keeping operation tracking internal to the object. - Considerations: Lacks type validation, so improper inputs could affect runtime behavior unless externally checked. Overall, the method provides a simple yet extensible pattern for mathematical operations with built-in traceability.
    """


        }
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessagePromptTemplate.from_template("{output}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt= example_prompt,
        examples= examples
    )

    code_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        few_shot_prompt,
        HumanMessagePromptTemplate.from_template(query)
    ])
    
    pipeline = (
        {
            "code_text": lambda x: x["code_text"],
            "user_query": lambda x: x["user_query"]

        }
        | code_prompt
        | llm
        )
    ai_message = pipeline.invoke(code_dict)
    return ai_message.content

    
    
