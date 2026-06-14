from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# ── System prompt template ────────────────────────────────────────────
# Variables injected at runtime:
#   {disease}    → detected class name e.g. "Potato___Early_blight"
#   {confidence} → e.g. "94.3"
#   {all_scores} → e.g. "Early Blight: 94.3%, Healthy: 3.1%, Late Blight: 2.6%"
# {history} is injected automatically by ConversationBufferMemory

SYSTEM_TEMPLATE = """You are PlantScan AI, an expert plant pathologist assistant embedded \
in a potato disease detection app. You are helpful, concise, and accurate.

You specialise in:
- Potato plant diseases (Early Blight, Late Blight)
- Fungal and bacterial infections in crops
- Treatment plans, fungicide recommendations
- Prevention and crop management best practices

Current scan result from the AI model:
  Detected condition : {disease}
  Confidence         : {confidence}%
  All class scores   : {all_scores}

Use this scan context to give specific, relevant answers. If the user asks \
"how do I treat it?" or "what caused this?" — answer based on the detected disease above.
Keep answers practical and to the point. Avoid unnecessary filler.
If no scan has been done yet, answer generally and encourage the user to scan a leaf first."""

def build_prompt() -> ChatPromptTemplate:
    """
    Builds and returns the full ChatPromptTemplate.
    MessagesPlaceholder with variable_name='history' is where
    ConversationBufferMemory automatically inserts past messages.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),   # ← memory goes here
        HumanMessagePromptTemplate.from_template("{input}")
    ])
