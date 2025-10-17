import os
import json
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from langchain.tools import tool

from app_5_RAG import master_rag, master_rag_tool


# ==========================================================
# 🔧 ENV + CONFIG
# ==========================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-5-nano"  # your cheapest JSON-capable model

TICKET_CATEGORIES = [
    "Make New Package",
    "Publish Artwork to Platform",
    "Change Existing Image Assets",
    "Add Missing Image Assets",
]
CONF_THRESHOLD = 60

USER_UNSURE = "I'm Unsure - Described in Request Section"

CHAT_RESPONSE_PARSER_CATEGORIES = [
    "yes",           # user agrees to change request type
    "no",            # user wants to keep current request type
    "unrelated",     # message unrelated to SST imagery
    "question",      # user asks a question about SST imagery
    "more context"   # user provides more ticket detail
]

# ==========================================================
# 🔧 LLM SETUP
# ==========================================================
def get_llm_json():
    """Return a ChatOpenAI client that *must* output JSON."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

@tool
def master_rag_tool(query: str) -> str:
    """Retrieve relevant context via RAG."""
    RAG_result = master_rag(query)
    return RAG_result.get("RAG_combined_text", "")

def get_llm_with_tools():
    """
    Returns a direct callable LLM that can access RAG context.
    This avoids 'stop' parameter errors and LangChain's deprecated initialize_agent.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    def call_with_rag(user_message: str):
        # 1️⃣ Get context via RAG
        rag_context = master_rag(user_message).get("RAG_combined_text", "")

        # 2️⃣ Build prompt manually (no 'stop' arg here)
        system_prompt = """
        You are a JSON-only assistant that can answer user questions or classify messages.
        Use ONLY the provided RAG context if relevant.
        """
        user_prompt = f"""
        User Message: {user_message}
        RAG Context:
        {rag_context}
        """

        # 3️⃣ Invoke directly on the ChatOpenAI model (no initialize_agent)
        response = llm.invoke([
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ])
        return (response.content or "").strip()

    return call_with_rag

# ==========================================================
# ⚙️ MASTER LLM CLASS
# ==========================================================
@dataclass
class MasterLLMResult:
    """Container for final structured LLM outputs."""
    result: str
    result_confidence: int
    result_explanation: str
    result_summary: str
    RAG_combined_title: str
    RAG_combined_total_chunk_number: int
    is_first_pass: bool

# ==========================================================
# 🤝 HELPERS
# ==========================================================
def request_type_summary_llm(request_type: str) -> str:
    """
    LLM that summarizes request_type based on its RAG context.
    """
    RAG_result = master_rag(request_type)
    rag_text = RAG_result["RAG_combined_text"]
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    system_prompt = "You are an expert text summarizer assistant. Use ONLY the RAG context."
    user_prompt = f"Define '{request_type}' in 1–2 concise sentences based ONLY on this context:\n{rag_text}"

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return (response.content or "").strip()


# ==========================================================
# 🧩 MASTER FUNCTION
# ==========================================================
def master_llm(request_description) -> MasterLLMResult:
    """
    Master LLM that provides all the required variables in the MasterLLMResult class
    """
    # --- Step 1: Perform RAG ---
    RAG_result = master_rag(request_description)
    RAG_combined_text = RAG_result["RAG_combined_text"]
    RAG_combined_title = RAG_result["RAG_combined_title"]
    RAG_combined_total_chunk_number = RAG_result.get("RAG_combined_total_chunk_number", 0)

    # --- Step 2: Prepare LLM, System, and User Prompts ---
    llm = get_llm_json()

    system_prompt = f"""
    You are a strict JSON-only classifier for ticket requests.
    Use ONLY the provided RAG context to determine which category best fits the request description.
    
    Categories (must pick EXACTLY one):
    {TICKET_CATEGORIES}

    Respond ONLY in valid JSON, like:
    {{
        "result": "Make New Package",
        "result_confidence": 87,
        "result_explanation": "Reasoning based on context",
        "result_summary": "Concise 1–2 sentence definition of the meaning of the result category"
    }}
    """

    user_prompt = f"""
    Request Description: {request_description}
    RAG CONTEXT: {RAG_combined_text}
    """

    # --- Step 3: Run LLM ---
    response = llm.invoke([
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ])
    raw = (response.content or "").strip()

    # --- Step 4: Ensures that the output is in the correct JSON format we want ---
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception: # this gives a fallback dictionary to prevent from crashing
        data = {
            "result": "LLM JSON ERROR",
            "result_confidence": 0,
            "result_explanation": "Parsing failed or no valid JSON output.",
            "result_summary": ""
        }

    # --- Step 5: Ensure the LLM actually used 1 of the categories defined ---
    result = data.get("result", "").strip()
    if result not in TICKET_CATEGORIES:
        result = "LLM CATEGORIZATION ERROR"
    
    # --- Step 6: Add first pass logic ---
    if "FIRST_PASS_DONE" not in st.session_state:
        st.session_state.FIRST_PASS_DONE = False

    is_first_pass = not st.session_state.FIRST_PASS_DONE
    st.session_state.FIRST_PASS_DONE = True

    return MasterLLMResult(
        result=result,
        result_confidence=int(data.get("result_confidence", 0)),
        result_explanation=data.get("result_explanation", ""),
        result_summary=data.get("result_summary", ""),
        RAG_combined_title=RAG_combined_title,
        RAG_combined_total_chunk_number=RAG_combined_total_chunk_number,
        is_first_pass=is_first_pass
    )

# ==========================================================
# 🧠 MASTER FLOW ORCHESTRATOR
# ==========================================================

def master_flow_orchestrator(master_result: MasterLLMResult, request_type: str, request_description: str):
    """
    Determine next step in LLM flow based on master_llm output.
    Returns structured dictionary for Streamlit or orchestration logic.
    """

    # --- Preset: Extract key variables from masterllmresult ---
    result = master_result.result
    result_confidence = master_result.result_confidence
    result_summary = master_result.result_summary
    is_first_pass = master_result.is_first_pass
    RAG_combined_total_chunk_number = master_result.RAG_combined_total_chunk_number

    # --- Handle Error Conditions ---

    if RAG_combined_total_chunk_number == 0:
        return {
            "flow_type": "RAG_RETRIEVAL_ERROR",
            "response_text": (
                "I wasn’t able to find any relevant context to classify this request. "
                "Could you please describe your ticket again or add a bit more detail?"
            ),
            "debug": "RAG retrieved 0 chunks",
            "action": "RETRY"
        }

    elif result == "LLM JSON ERROR":
        return {
            "flow_type": "LLM_JSON_ERROR",
            "response_text": (
                "I couldn’t confidently determine the request type based on your description. "
                "Could you please describe your ticket again or add a bit more detail?"
            ),
            "debug": "The LLM responded in a format outside of JSON requirements.",
            "action": "RETRY"
        }

    elif result == "LLM CATEGORIZATION ERROR":
        return {
            "flow_type": "LLM_CATEGORIZATION_ERROR",
            "response_text": (
                "I couldn’t confidently determine the request type based on your description. "
                "Could you please describe your ticket again or add a bit more detail?"
            ),
            "debug": "The LLM gave an undefined category.",
            "action": "RETRY"
        }

    elif result_confidence <= 60:
        return {
            "flow_type": "LOW_CONFIDENCE_ERROR",
            "response_text": (
                "I couldn’t confidently determine the request type based on your description. "
                "Could you please describe your ticket again or add a bit more detail?"
            ),
            "debug": "confidence level of result is <= 60",
            "action": "RETRY"
        }

    # --- Match / Correct Path ---

    elif result == request_type: #VERY IMPORTANT INFO - If it doesn't go through the first time, and the user ends up describing the problem in a way that matches the type, it will immediately go through
        return {
            "flow_type": "CORRECT_MATCH_COMPLETE",
            "response_text": "✅ It looks like your request went through correctly!",
            "debug": "result matches request type",
            "action": "COMPLETE"
        }

    # --- Step 4: More Context Path ---
    elif not is_first_pass:
        more_context_msg = (
            f"Based on the new context you provided, "
            f"I think '{result}' is the request type you're looking for.\n\n"
            f"'{result}' means: {result_summary}\n\n"
            f"Would you like me to change the Request Type to '{result}'?"
        )
        return {
            "flow_type": "MORE_CONTEXT_LLM",
            "response_text": more_context_msg,
            "action": "CHATLLM"
        }

    # --- Step 5: Unsure Path (First Pass) ---
    elif request_type == "I'm Unsure - Described in Request Section" and is_first_pass:
        unsure_msg = (
            f"It looks like you weren't sure which request type your request falls under. "
            f"No worries, I will help you out.\n\n"
            f"Based on your Request Description '{request_description}', "
            f"I think '{result}' is the request type you're looking for.\n\n"
            f"'{result}' means: {result_summary}\n\n"
            f"Would you like me to update the Request Type to '{result}' or choose another choice?"
        )
        return {
            "flow_type": "USER_UNSURE_LLM",
            "response_text": unsure_msg,
            "action": "CHATLLM"
        }

    # --- Step 6: Mismatch Path (First Pass) ---
    elif request_type != "I'm Unsure - Described in Request Section" and is_first_pass:
        # Call request_type_summary_llm
        request_type_summary = request_type_summary_llm(request_type)

        mismatch_msg = (
            f"It looks like you chose '{request_type}' for this ticket. "
            f"However, based on your Request Description '{request_description}', "
            f"I think '{result}' is the request type you're actually looking for.\n\n"
            f"'{request_type}' means: {request_type_summary}\n\n"
            f"'{result}' means: {result_summary}\n\n"
            f"Would you like me to change the Request Type to '{result}' or continue with your previous choice?"
        )
        return {
            "flow_type": "MISMATCH_LLM",
            "response_text": mismatch_msg,
            "action": "CHATLLM"
        }


# ==========================================================
# 💬 USER CHAT CLASSIFIER (for page 2)
# ==========================================================
def chat_response_parser_llm(user_message: str, RAG_context_text: str = "") -> dict:
    """
    Classifies user's chat message into one of five categories:
    'yes', 'no', 'unrelated', 'question', or 'more context'.
    Returns a JSON dict: { "user_response_type": "...", "response": "..." }
    """
    # --- Step 1: Initialize JSON-mode LLM ---
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    # --- Step 2: System + user prompts ---
    system_prompt = f"""
    You are a strict JSON-only classifier for user chat responses.
    You must choose ONE AND ONLY ONE of these valid categories for "user_response_type":
    {CHAT_RESPONSE_PARSER_CATEGORIES}

    Definitions:
    - yes → the user explicitly agrees or approves the suggested change in request type 
            (e.g., "yes", "ok", "sure", "please change it", "sounds good", "go ahead").
    - no → the user explicitly disagrees or rejects the suggested change 
            (e.g., "no", "keep as is", "don't change it", "leave it").
    - unrelated → the message is unrelated to SkyShowtime imagery or the ticket system 
                (e.g., talking about weather, movies, or jokes).
    - question → the user is asking a question about SST imagery or the process.
    - more context → the user is adding or clarifying information about the ticket content 
                    (for example: adding details about episodes, localizations, or imagery).

    🔹 Classification priority:
    1️⃣ If the message contains “yes” or “no” but those words are part of a longer sentence that
        adds information about the *ticket content* (episodes, assets, imagery, etc.),
        classify as **more context**, not “yes” or “no”.
    2️⃣ If the overall message semantic intent is about agreeing or rejecting a change, 
        classify as **yes** or **no** — even if the user adds small clarifying words like “please” or “for me”.
    3️⃣ If the “yes/no” appears only as part of a descriptive or clarifying message about the ticket content, classify as **more context**.
    4️⃣ If the message both gives context and expresses a decision, base the classification
        on the *main intent*:  
        • If the main goal is to approve/deny → **yes/no**  
        • If the main goal is to explain or describe → **more context**

    Examples:
    - “Yes, please change it for me.” → yes  
    - “No, keep the same type.” → no  
    - “No, I meant episodes 3–7 are missing.” → more context  
    - “Please add episodes 3–7 to the package.” → more context  
    - “Can you explain what request type means?” → question  
    - “What’s for lunch?” → unrelated  

    Respond ONLY in valid JSON format like:
    {{
        "user_response_type": "yes"
    }}
    """


    user_prompt = f"""
    User Message: {user_message}
    Optional Context: {RAG_context_text}
    """

    # --- Step 3: Run the LLM directly ---
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ])
        raw = (response.content or "").strip()
    except Exception as e:
        raw = json.dumps({
            "user_response_type": "unrelated",
            "response": f"Error occurred during classification: {e}"
        })

    # --- Step 4: Parse JSON safely ---
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception:
        data = {
            "user_response_type": "unrelated",
            "response": "Sorry, I couldn’t process that message properly."
        }

    # --- Step 5: Enforce valid category ---
    user_response_type = str(data.get("user_response_type", "")).lower().strip()
    if user_response_type not in CHAT_RESPONSE_PARSER_CATEGORIES:
        user_response_type = "unrelated"

    # --- Step 6: Handle 'question' category using RAG ---
    if user_response_type == "question":
        RAG_result = master_rag(user_message)
        RAG_context = RAG_result.get("RAG_combined_text", "")

        llm_answer = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
        system_answer = """
        You are a SkyShowtime imagery production assistant.
        Answer the user's question concisely using the provided RAG context below.
        If the RAG context doesn’t contain the information, say:
        "Sorry, I couldn't find details about that information."
        """
        user_question = f"User Question: {user_message}\n\nRAG Context:\n{RAG_context}"

        try:
            resp = llm_answer.invoke([
                {"role": "system", "content": system_answer.strip()},
                {"role": "user", "content": user_question.strip()}
            ])
            data["response"] = (resp.content or "").strip()
        except Exception as e:
            data["response"] = f"Sorry, I couldn't retrieve information right now. ({e})"

    # --- Step 7: Handle unrelated messages politely ---
    if user_response_type == "unrelated":
        data["response"] = "I'm sorry, but I can only assist with SkyShowtime imagery production requests."

    # --- Step 8: Return structured result ---
    return {
        "user_response_type": user_response_type,
        "response": data.get("response", "")
    }





# ==========================================================
# 🧪 Manual Test
# ==========================================================


# --- Interactive MasterLLM Testing ---
#if __name__ == "__main__":
#    while True:
#        user_input = input("\n📝 Enter Request Description (or 'q' to quit): ").strip()
#        if user_input.lower() in ["q", "quit", "exit"]:
#            break

#        llmresult = master_llm(user_input)
#        result = master_flow_orchestrator(llmresult)

 #       print("\n=== Master LLM Result ===")
  #      print(f"Result: {result.result}")
   #     print(f"Confidence: {result.result_confidence}")
    #    print(f"Explanation: {result.result_explanation}")
     #   print(f"Summary: {result.result_summary}")
      #  print(f"RAG Titles: {result.RAG_combined_title}")
       # print(f"Total Chunks: {result.RAG_combined_total_chunk_number}")
        #print(f"Is First Pass: {result.is_first_pass}")


# --- Interactive chat_response_parser_llm Testing --- 
if __name__ == "__main__":
    user_input = input("\n📝 Enter Request Description (or 'q' to quit): ").strip()
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 Exiting test mode.")
        import sys
        sys.exit()

    
    result = chat_response_parser_llm(user_input)

    print("\n=== Chat Parser LLM Result ===")
    print(f"user_response_type: {result['user_response_type']}")
    print(f"response: {result['response']}")


# --- Interactive MasterLLM + MasterOrchestrator Testing --- 
#if __name__ == "__main__":
#    request_type = "hello" # adding request type variable here for testing
#
#    print("🔹 SST Ticketing Agent — Interactive Test Mode 🔹")
#    print("Type a request description below. Type 'q' to quit.\n")
#    print(f"{request_type}")

 #   while True:
        # --- Step 1: get user input ---
  #      user_input = input("\n📝 Enter Request Description (or 'q' to quit): ").strip()
   #     if user_input.lower() in ["q", "quit", "exit"]:
    #        print("👋 Exiting test mode.")
     #       break

        # --- Step 2: Run the master LLM on the input ---
      #  master_llm_result = master_llm(user_input)

        # --- Step 3: Run orchestration logic ---
       # master_flow_result = master_flow_orchestrator(master_llm_result, request_type, user_input)

        # --- Step 4: Print which branch fired ---
       # print("\n==================== FLOW RESULT ====================")
        #print(f"🧩 Flow Type: {master_flow_result['flow_type']}")
       # print(f"💬 Response Text:\n{master_flow_result['response_text']}")
        #print(f"⚙️ Action: {master_flow_result.get('action', '(none)')}")
       # print(f"🐞 Debug: {master_flow_result.get('debug', '(none)')}")
        #print("=====================================================\n")

        # --- Step 5 (optional): show internal LLM variables ---
     #   print("🔍 INTERNAL VARIABLES:")
      #  print(f"Result: {master_llm_result.result}")
    #    print(f"Confidence: {master_llm_result.result_confidence}")
      #  print(f"Summary: {master_llm_result.result_summary}")
     #   print(f"Is First Pass: {master_llm_result.is_first_pass}")
      #  print(f"RAG Chunk Count: {master_llm_result.RAG_combined_total_chunk_number}")
     #   print("-----------------------------------------------------\n")
