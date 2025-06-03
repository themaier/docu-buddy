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


def get_business_qa(code_dict: dict) -> str:
    
    system_prompt = """
    You are a smart, detail-oriented code explanation assistant specialized in simplifying complex technical information for business stakeholders.

    You will be given two types of inputs:
    1. A block of code (code input) — your job is to analyze it carefully and explain it in a way that is **easy for non-technical business professionals** to understand. Focus on **what** the code does, **why** it’s important, and **how** it supports business goals — **not** how it works technically.
    2. A query — a natural language question from the user about a specific part or behavior of the previously submitted code.

    Your tasks are:
    - When code is provided, **summarize** its purpose, main components, and high-level flow in simple, business-friendly terms. Avoid technical jargon. Focus on benefits, outcomes, and business value.
    - When a query is provided, respond only based on the code given, offering a **clear, non-technical explanation** of the relevant part, focusing on **impact**, **purpose**, and **practical meaning** instead of deep technical detail.

    Your responses should:
    - Be easy to read and free of complex programming language.
    - Relate the code’s function to **business needs** like efficiency, cost savings, customer experience, scalability, or risk reduction.
    - Explain the **"what"** and **"why"** without getting lost in the **"how."**
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
    Based on the following inputs, analyze the provided code and answer the user’s query with a detailed explanation.

    Inputs:
    - Code Text:
    def add_numbers(a, b):
        return a + b

    - User Query:
    What does the add_numbers function do?
    """,
            "output": """
    The add_numbers function is a simple tool designed to combine two pieces of information by adding them together.

    - **What it does**: Takes two inputs (for example, two numbers) and gives back their total.
    - **Why it's useful**: Helps quickly calculate totals — like adding up order amounts, scores, or financial figures — without manual effort.
    - **What it expects**: The information provided should be compatible — meaning it should make sense to add them (like two numbers or two words).
    - **Design approach**: It’s built to be simple and fast, focusing only on adding without checking if the inputs are correct.
    - **Things to keep in mind**: If the wrong type of information is given (for example, a number and a word), it could cause an error. Usually, other parts of the system ensure this doesn’t happen.

    **In short**: It’s a basic but essential function that makes adding two values easy and efficient — a small building block often used in larger business applications like billing systems or performance tracking.
    """

        },
        {
            "input": """
    Based on the following inputs, analyze the provided code and answer the user’s query with a detailed explanation.

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

    ```class Calculator:
        def __init__(self):
            self.history = []

        def multiply(self, x, y):
            result = x * y
            self.history.append(('multiply', x, y, result))
            return result```

    The multiply method in the Calculator class is designed to multiply two values and keep a record of each calculation for future reference.

    - **What it does**: Takes two numbers (like 6 and 7), multiplies them, and gives you the result (42).
    - **Keeping records**: After every multiplication, it saves the details — what was multiplied and what the result was — into a list, like keeping a mini history or logbook.
    - **Why this matters**: This history lets users review past calculations without having to redo them, which is helpful for generating reports or verifying work.
    - **Smart design**: Everything related to the calculation and the record-keeping stays neatly within the calculator, making it organized and easy to manage.
    - **Important note**: It assumes the inputs are correct (for example, that the user is entering numbers). If the wrong type of information is entered, it could cause problems, so usually other parts of the system help with checking inputs.

    **In short**: This method doesn’t just calculate — it also builds a trusted record of all past work, which can improve reliability, transparency, and accountability in business tasks like financial reports or operational audits.
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

    
    
