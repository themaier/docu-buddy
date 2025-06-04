import os
import sys
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    complexity_analyzer,
    download_github_repo,
    llm_complexity_analyzer,
    supabase_access,
)
from pdf_handling import pdf
from repo_checker import tll
from repo_checker import matrix
# from business_QA import get_business_qa
# from developer_QA import get_developer_qa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="My API",
    description="A simple FastAPI app deployed on Railway",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Developer(BaseModel):
    user_query: str


class GitHubRepoRequest(BaseModel):
    url: HttpUrl


# Routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to my FastAPI app!",
        "status": "running",
        "endpoints": [
            "/docs - API documentation",
            "/items - Get all items",
            "/items/{id} - Get specific item",
            "/items (POST) - Create new item",
        ],
    }
    

@app.post("/upload/tll")
async def upload_to_tll(file: UploadFile = File(...)):
    result = pdf.process_pdf_and_upload(file.file, table="tll")
    return result

@app.post("/upload/matrix")
async def upload_to_matrix(file: UploadFile = File(...)):
    result = pdf.process_pdf_and_upload(file.file, table="matrix")
    return result

@app.get("/check/tll")
async def check_tll_compliance():
    """
    Scan ./repo, decide which languages / frameworks are permitted
    according to the Technische Vorgaben PDF stored in the 'tll' table.
    """
    try:
        link_map = tll.list_excluded_files_with_links("./repo", include_ignored=False)
        tll.insert_supabase(link_map)
        return json.dumps(link_map, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    

@app.get("/check/matrix")
async def check_matrix_compliance():
    """
    Scan ./repo, decide which languages / frameworks are permitted
    according to the Technische Vorgaben PDF stored in the 'tll' table.
    """
    try:
        # result = matrix.check_repo("./repo")
        result = ""
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/download-repo")
def download_repo(payload: GitHubRepoRequest):
    try:
        url = str(payload.url).rstrip("/")
        if not url.startswith("https://github.com/"):
            raise ValueError("Invalid GitHub URL format")
        dest_path = download_github_repo.download_github_repo_zip(url)
        complexity_analyzer.main(repo_url=f"{url}/blob/main/")
        llm_complexity_analyzer.main()
        supabase_access.upload_function_complexity()

        return {"message": "Repository analised successfully", "path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
@app.post("/check-repo")
def check_repo(payload: GitHubRepoRequest):
    try:
        url = str(payload.url).rstrip("/")
        if not url.startswith("https://github.com/"):
            raise ValueError("Invalid GitHub URL format")
        dest_path = download_github_repo.download_github_repo_zip(url)
        complexity_analyzer.main(repo_url=f"{url}/blob/main/")
        llm_complexity_analyzer.main()
        supabase_access.upload_function_complexity()

        return {"message": "Repository analised successfully", "path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


@app.post("/embedd-repo")
def embedd_repo(payload: GitHubRepoRequest):
    try:
        url = str(payload.url).rstrip("/")
        if not url.startswith("https://github.com/"):
            raise ValueError("Invalid GitHub URL format")
        dest_path = download_github_repo.download_github_repo_zip(url)
        complexity_analyzer.main(repo_url=f"{url}/blob/main/")
        llm_complexity_analyzer.main()
        supabase_access.upload_function_complexity()

        return {"message": "Repository analised successfully", "path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# @app.post("/developer", status_code=status.HTTP_201_CREATED)
# def get_developer_response(user_query: Developer) -> str:
#     """
#     Simulated function to analyze code and respond to queries.
#     In a real application, this would call an LLM or other analysis tool.
#     """
#     # Placeholder response
#     # the relevant code text should be queried from the vector database using RAG based on the user_query
#     code_text = """
#     class BankAccount:
#         def __init__(self, account_holder, balance=0):
#             self.account_holder = account_holder
#             self.balance = balance

#         def deposit(self, amount):
#             if amount > 0:
#                 self.balance += amount
#                 return True
#             else:
#                 return False

#         def withdraw(self, amount):
#             if 0 < amount <= self.balance:
#                 self.balance -= amount
#                 return True
#             else:
#                 return False
#     """

#     response = get_developer_qa({"code_text": code_text, "user_query": user_query})
#     if not response:
#         raise HTTPException(status_code=400, detail="Invalid query or code text")
#     return response


# @app.post("/business", status_code=status.HTTP_201_CREATED)
# def get_developer_response(user_query: Developer) -> str:
#     """
#     Simulated function to analyze code and respond to queries.
#     In a real application, this would call an LLM or other analysis tool.
#     """
#     # Placeholder response
#     # the relevant code text should be queried from the vector database using RAG based on the user_query
#     code_text = """
#     class BankAccount:
#         def __init__(self, account_holder, balance=0):
#             self.account_holder = account_holder
#             self.balance = balance

#         def deposit(self, amount):
#             if amount > 0:
#                 self.balance += amount
#                 return True
#             else:
#                 return False

#         def withdraw(self, amount):
#             if 0 < amount <= self.balance:
#                 self.balance -= amount
#                 return True
#             else:
#                 return False
#     """

#     response = get_business_qa({"code_text": code_text, "user_query": user_query})
#     if not response:
#         raise HTTPException(status_code=400, detail="Invalid query or code text")
#     return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
