from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from chat.memory import get_or_create_memory

def get_llm(model: str = "llama3.2") -> OllamaLLM:
    return OllamaLLM(
        model=model,
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=512,
    )

def build_chain(session_id: str, model: str = "llama3.2") -> RunnableWithMessageHistory:
    llm = get_llm(model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are PlantScan AI, an expert plant pathologist assistant.
Current scan: {disease} detected with {confidence}% confidence.
All scores: {all_scores}
Answer specifically based on this scan result. Be concise and practical."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_or_create_memory,
        input_messages_key="input",
        history_messages_key="history",
    )