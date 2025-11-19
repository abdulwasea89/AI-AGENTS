import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from PyPDF2 import PdfReader
from docx import Document
from langchain.tools import tool
from googlesearch import search
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(doc_path: str) -> str:
    doc = Document(doc_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@tool
def resume_parser(resume_file_path: str) -> str:
    """Parse a resume file (.pdf or .docx) and return extracted text."""
    if resume_file_path.endswith(".pdf"):
        return extract_text_from_pdf(resume_file_path)
    elif resume_file_path.endswith(".docx"):
        return extract_text_from_docx(resume_file_path)
    else:
        raise ValueError("Unsupported file type")

@tool
def google_search(query: str) -> list:
    """Perform a Google search and return top results."""
    return list(search(query, num_results=5))

@tool
def general_question_answer(question: str) -> str:
    """Answer a general question using the model."""
    response = model.invoke(question)
    return response.content

resume_parser_agent = create_react_agent(
    model,
    tools=[resume_parser],
    name="resume_parser_agent",
    prompt=(
        "You are a resume parser expert. "
        "Always use the one tool resume_parser to parse the resume."
    )
)

google_search_agent = create_react_agent(
    model,
    tools=[google_search],
    name="google_search_agent",
    prompt=(
        "You are a Google search expert. "
        "Always use the one tool google_search to search the internet."
    )
)

general_question_answer_agent = create_react_agent(
    model,
    tools=[general_question_answer],
    name="general_question_answer_agent",
    prompt=(
        "You are a general question answer expert. "
        "Always use the one tool general_question_answer to answer the question."
    )
)

workflow = create_supervisor(
    [resume_parser_agent, google_search_agent, general_question_answer_agent],
    model=model,
    prompt=(
        "You are a smart team supervisor managing multiple agents. Analyze the user input and delegate to the appropriate agent:\n"
        "- If the input contains a file path or mentions 'resume', use resume_parser_agent.\n"
        "- If the input contains 'search' or asks to find something online, use google_search_agent.\n"
        "- For all other questions or queries, use general_question_answer_agent.\n"
        "Choose the most appropriate agent based on the user's input."
    ),
    output_mode="last_message"
)

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    result = app.invoke({
        "messages": [{
            "role": "user",
            "content": user_input
        }]
    }, config=config)
    for m in result["messages"]:
        print(m.content)
