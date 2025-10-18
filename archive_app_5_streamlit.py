# ==========================================================
# app_streamlit_page1.py — SST Ticket • Step 1
# ==========================================================
import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

from app_5_LLM import master_llm, master_flow_orchestrator, chat_response_parser_llm

# ==========================================================
# 🔐 ENV
# ==========================================================
load_dotenv()
AIRTABLE_API_KEY             = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID             = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_MAIN_TABLE_ID       = os.getenv("AIRTABLE_MAIN_TABLE_ID")
AIRTABLE_REQUESTORS_TABLE_ID = os.getenv("AIRTABLE_REQUESTORS_TABLE_ID")
AIRTABLE_WATCHLIST_TABLE_ID  = os.getenv("AIRTABLE_WATCHLIST_TABLE_ID")

HEADERS = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
AIRTABLE_BASE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
AIRTABLE_META_URL = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"

# ==========================================================
# ✅ REQUIRED FIELDS
# ==========================================================
MANDATORY = {
    "Submitted Team Branch": False,
    "Requested By": False,
    "Watch List": False,
    "Content Brand": False,
    "Content Title": False,
    "Content Type": False,
    "Season #": False,
    "Episode #": False,
    "Request Type": True,
    "Request Description": True,
}

# ==========================================================
# 🔧 Airtable Helpers
# ==========================================================
@st.cache_data
def get_select_options(field_name: str):
    """Fetch singleSelect/multipleSelect options for a field."""
    try:
        r = requests.get(AIRTABLE_META_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        meta = r.json()
        for t in meta.get("tables", []):
            if t.get("id") == AIRTABLE_MAIN_TABLE_ID:
                for f in t.get("fields", []):
                    if f.get("name") == field_name and f.get("type") in ("singleSelect", "multipleSelects"):
                        return [c["name"] for c in f["options"]["choices"]]
        return []
    except Exception as e:
        st.error(f"Error fetching options for '{field_name}': {e}")
        return []

@st.cache_data
def get_linked_record_options(table_id: str, display_field="Name"):
    """Return list of linked records [{id, name}]."""
    try:
        r = requests.get(f"{AIRTABLE_BASE_URL}/{table_id}", headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return [
            {"id": rec["id"], "name": rec.get("fields", {}).get(display_field, f"Unnamed ({rec['id']})")}
            for rec in data.get("records", [])
        ]
    except Exception as e:
        st.error(f"Error fetching linked records: {e}")
        return []

def create_main_record(fields: dict):
    try:
        r = requests.post(
            f"{AIRTABLE_BASE_URL}/{AIRTABLE_MAIN_TABLE_ID}",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"records": [{"fields": fields}]},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return data["records"][0]["id"]
    except Exception as e:
        st.error(f"Error creating record: {e}")
        return None

def update_record(record_id: str, fields: dict):
    """Update an existing Airtable record by ID."""
    try:
        r = requests.patch(
            f"{AIRTABLE_BASE_URL}/{AIRTABLE_MAIN_TABLE_ID}/{record_id}",
            headers={**HEADERS, "Content-Type": "application/json"},
            json={"fields": fields},
            timeout=20,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error updating record: {e}")
        return False

# ==========================================================
# 🧠 DEBUG TRACE HELPER
# ==========================================================
def show_llm_trace(parse_result=None, master_llm_result=None, master_flow_result=None) -> str:
    """
    Returns a formatted Markdown string for debugging (does NOT render directly).
    Each assistant message stores this string so it's only displayed under that message.
    """
    debug_info = []

    # Chat parser results
    if parse_result:
        debug_info.append(
            f"\n"
            f"🟦 **chat_response_parser_llm:** "
            f"user_response_type=`{parse_result.get('user_response_type', '-')}`"
        )

    # Master LLM results
    if master_llm_result:
        debug_info.append(
            f"\n"
            f"🟩 **master_llm:** "
            f"result=`{getattr(master_llm_result, 'result', '-')}`, "
            f"confidence=`{getattr(master_llm_result, 'result_confidence', '-')}%`, "
            f"summary=`{getattr(master_llm_result, 'result_summary', '-')}`"
        )

    # Orchestrator results
    if master_flow_result:
        debug_info.append(
            f"\n"
            f"🟨 **master_flow_orchestrator:** "
            f"flow_type=`{master_flow_result.get('flow_type', '-')}`, "
            f"action=`{master_flow_result.get('action', '-')}`"
        )

    # Combine and return (no Streamlit calls here!)
    if debug_info:
        return "\n".join(debug_info)
    return ""


# ==========================================================
# 🔧 UI Helpers (Asterisk)
# ==========================================================
def render_label(label: str):
    star = " <span style='color:#e00'>&nbsp;*</span>" if MANDATORY.get(label, False) else ""
    st.markdown(f"<div style='font-weight:600; margin:2px 0 4px 0;'>{label}{star}</div>", unsafe_allow_html=True)

def labeled_selectbox(label, options, key):
    render_label(label)
    return st.selectbox("", options, key=key, label_visibility="collapsed")

def labeled_multiselect(label, options, key):
    render_label(label)
    return st.multiselect("", options, key=key, label_visibility="collapsed")

def labeled_text_input(label, key):
    render_label(label)
    return st.text_input("", key=key, label_visibility="collapsed")

def labeled_text_area(label, key, **kwargs):
    render_label(label)
    return st.text_area("", key=key, label_visibility="collapsed", **kwargs)

# ==========================================================
# Request description documentation helper
# ==========================================================

def update_consolidated_request_description(record_id: str):
    """
    Combine all user inputs and the original request_description
    into one long text, then update Airtable's "Request Description" field.
    Order: newest → oldest → original.
    """
    try:
        # Start from the current session data
        all_user_inputs = [
            msg["content"] for msg in st.session_state.get("chat_history", [])
            if msg.get("role") == "user"
        ][::-1]
        original_description = st.session_state.get("original_request_description", "")

        # Concatenate (newest first)
        combined_text = "\n\n".join(reversed(all_user_inputs + [original_description]))

        # Update Airtable record
        fields = {"Request Description": combined_text}
        success = update_record(record_id, fields)

        if success:
            st.info("🪶 Updated Airtable Request Description with all chat context.")
        else:
            st.warning("⚠️ Failed to update Airtable Request Description.")
    except Exception as e:
        st.error(f"Error updating consolidated Request Description: {e}")

# ==========================================================
# ALL chat documentation helper
# ==========================================================

def update_chat_history_record(record_id: str):
    """
    Combine the entire chat_history (assistant + user + debug)
    and save it to the Airtable 'Chatbot Chat History' field.
    """
    try:
        chat_entries = []
        for msg in st.session_state.get("chat_history", []):
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            debug = msg.get("debug", "")
            if debug:
                chat_entries.append(f"{role}: {content}\n{debug}")
            else:
                chat_entries.append(f"{role}: {content}")

        # Combine with spacing
        full_chat_log = "\n\n".join(chat_entries)

        # Update Airtable
        fields = {"Chatbot Chat History": full_chat_log}
        success = update_record(record_id, fields)

        if success:
            st.info("💾 Chat history successfully saved to Airtable.")
        else:
            st.warning("⚠️ Failed to update Chatbot Chat History.")
    except Exception as e:
        st.error(f"Error updating Chatbot Chat History: {e}")

# ==========================================================
# 🧭 PAGE ROUTING (keep all pages in this file)
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = 1  # Default start on Page 1

# ==============================================================================================================================================================================
# 🧾 PAGE 1 — Ticket Details
# ==============================================================================================================================================================================
def page_1_ticket_form():
    st.title("🎟️ SST IO AI Ticketing Agent")
    st.subheader("Please Enter Ticket Details")

    # --- Fetch Airtable options ---
    Submitted_Team_Branch = get_select_options("Submitted Team Branch")
    Content_Brand = get_select_options("Content Brand")
    Content_Type = get_select_options("Content Type")
    Season_Num = get_select_options("Season #")
    Episode_Num = get_select_options("Episode #")
    Request_Type = get_select_options("Request Type")
    Requested_By = get_linked_record_options(AIRTABLE_REQUESTORS_TABLE_ID)
    Requested_By_Names = [r["name"] for r in Requested_By]
    Watch_List = get_linked_record_options(AIRTABLE_WATCHLIST_TABLE_ID)
    Watch_List_Names = [w["name"] for w in Watch_List]

    # --- Widgets ---
    labeled_selectbox("Submitted Team Branch", Submitted_Team_Branch or ["(No options)"], "submitted_team_branch")
    labeled_selectbox("Requested By", Requested_By_Names or ["(No options)"], "requested_by_name")
    labeled_multiselect("Watch List", Watch_List_Names or ["(No options)"], "watchlist_names")
    labeled_selectbox("Content Brand", Content_Brand or ["(No options)"], "content_brand")
    labeled_text_input("Content Title", "content_title")
    labeled_selectbox("Content Type", Content_Type or ["(No options)"], "content_type")

    if st.session_state.get("content_type") == "Series":
        st.subheader("📺 Series Details")
        labeled_selectbox("Season #", Season_Num or ["(No options)"], "season_num")
        labeled_multiselect("Episode #", Episode_Num or ["(No options)"], "episode_nums")

    labeled_selectbox("Request Type", Request_Type or ["(No options)"], "request_type")
    labeled_text_area(
        "Request Description", 
        "request_description",
        height=200,
        placeholder="Please describe your request as detailed as possible and avoid ambiguous phrasing. Thank you!"
    )

    # --- Validate Required Fields ---
    def _is_missing(name, val):
        if not MANDATORY.get(name, False): return False
        if val is None or (isinstance(val, str) and not val.strip()) or (isinstance(val, list) and len(val) == 0):
            st.warning(f"⚠️ '{name}' is required.")
            return True
        return False

    required_now = [
        ("Submitted Team Branch", st.session_state.get("submitted_team_branch")),
        ("Requested By", st.session_state.get("requested_by_name")),
        ("Watch List", st.session_state.get("watchlist_names")),
        ("Content Brand", st.session_state.get("content_brand")),
        ("Content Title", st.session_state.get("content_title")),
        ("Content Type", st.session_state.get("content_type")),
        ("Request Type", st.session_state.get("request_type")),
        ("Request Description", st.session_state.get("request_description")),
    ]
    if st.session_state.get("content_type") == "Series":
        required_now += [("Season #", st.session_state.get("season_num")),
                        ("Episode #", st.session_state.get("episode_nums"))]

    disable_next = any(_is_missing(name, val) for name, val in required_now)



    # --- Next Button ---
    if st.button("Next →", type="primary", disabled=disable_next):
        # Step 1️⃣ Build the Airtable payload
        fields = {
            "Submitted Team Branch": st.session_state.get("submitted_team_branch"),
            "Requested By": [r["id"] for r in get_linked_record_options(AIRTABLE_REQUESTORS_TABLE_ID)
                            if r["name"] == st.session_state.get("requested_by_name")],
            "Watch List": [w["id"] for w in get_linked_record_options(AIRTABLE_WATCHLIST_TABLE_ID)
                        if w["name"] in st.session_state.get("watchlist_names", [])],
            "Content Brand": st.session_state.get("content_brand"),
            "Content Title": st.session_state.get("content_title"),
            "Content Type": st.session_state.get("content_type"),
            "Request Type": st.session_state.get("request_type"),
            "Request Description": st.session_state.get("request_description"),
        }

        # Setup Variables
        request_type = st.session_state.get("request_type")
        request_description = st.session_state.get("request_description")

        # Run LLM
        master_llm_result = master_llm(request_description)
        master_flow_result = master_flow_orchestrator(master_llm_result, request_type, request_description)

        if "original_request_description" not in st.session_state:
            st.session_state.original_request_description = st.session_state.get("request_description", "")


        #Create a new record in Airtable
        record_id = create_main_record(fields)

        #Save the new record ID in session
        if record_id:
            st.session_state.record_id = record_id
            # Step 4️⃣ Confirm success visually
            st.success(f"✅ Ticket successfully created in Airtable!")
            st.info(f"🆔 Record ID: `{record_id}`")
            st.json(fields)
        else:
            st.error("❌ Failed to create Airtable record. Please check your API key or network.")
        

        # determine page to route to
        if master_flow_result.get("action") == "COMPLETE":
            st.session_state.page = 3
            st.rerun()
        elif master_flow_result.get("action") in ["CHATLLM", "RETRY"]:
            st.session_state.master_flow_result = master_flow_result
            st.session_state.master_llm_result = master_llm_result
            st.session_state.page = 2
            st.rerun()








# ==============================================================================================================================================================================
# 🧾 PAGE 2 — LLM Chatbot
# ==============================================================================================================================================================================
def page_2_llm_chat():
    st.title("🎟️ SST IO AI Ticketing Agent")
    st.subheader("Ticket-Type AI Assistant")

    # --- Load session variables ---
    master_llm_result = st.session_state.get("master_llm_result", {})
    request_type = st.session_state.get("request_type")
    request_description = st.session_state.get("request_description")
    master_flow_result = st.session_state.get("master_flow_result", {})
    record_id = st.session_state.get("record_id", None)
    master_flow_orchestrator_action = master_flow_result.get("action", "")

    # --- Initialize chat history if empty ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

        first_msg = master_flow_result.get(
            "response_text",
            "Let's continue your ticket. Could you please describe your request again?"
        )

        # ✅ Create initial debug trace from the current LLM + flow state
        first_debug = show_llm_trace(None, master_llm_result, master_flow_result)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": first_msg,
            "debug": first_debug
        })


    # --- Display all previous chat messages ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("debug"):
                st.caption(msg["debug"])

    # --- Chat Input ---
    user_input = st.chat_input("Type your reply…")
    if not user_input:
        return

    # --- Add user message ---
    st.session_state.chat_history.append({"role": "user", "content": user_input, "debug": ""})
    with st.chat_message("user"):
        st.markdown(user_input)
        

    # ==========================================================
    # 🔁 Handle RETRY logic — re-run full LLM classification
    # ==========================================================
    if master_flow_orchestrator_action == "RETRY":
        request_description = user_input
        st.session_state.request_description = request_description

        master_llm_result = master_llm(request_description)
        master_flow_result = master_flow_orchestrator(master_llm_result, request_type, request_description)

        st.session_state.master_llm_result = master_llm_result
        st.session_state.master_flow_result = master_flow_result

        new_action = master_flow_result.get("action", "")
        new_response_text = master_flow_result.get("response_text", "I'm not sure how to proceed.")

        debug_text = show_llm_trace(None, master_llm_result, master_flow_result)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": new_response_text,
            "debug": debug_text
        })

        with st.chat_message("assistant"):
            st.markdown(new_response_text)
            if debug_text:
                st.caption(debug_text)

        st.session_state.master_flow_orchestrator_action = new_action

        if new_action == "COMPLETE":
            st.session_state.page = 3
            st.rerun()

    # ==========================================================
    # 💬 Handle CHATLLM logic — conversational follow-up
    # ==========================================================
    elif master_flow_orchestrator_action == "CHATLLM":
        parse_result = chat_response_parser_llm(user_input)
        user_response_type = parse_result.get("user_response_type", "").lower().strip()
        parser_response = parse_result.get("response", "")

        # === YES → Update Airtable and finish ===
        if user_response_type == "yes":
            latest_result = st.session_state.master_llm_result.result
            if not latest_result:
                st.warning("⚠️ No LLM result found. Please try again.")
                return

            try:
                success = update_record(st.session_state.record_id, {"Request Type": latest_result})
                if success:
                    st.success(f"✅ Airtable updated: Request Type → {latest_result}")
                else:
                    st.warning("⚠️ Could not update Airtable record.")
            except Exception as e:
                st.warning(f"Could not update Airtable: {e}")

            confirmation_msg = f"✅ Done — I’ve updated the Request Type to **{latest_result}** in your ticket."
            debug_text = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": confirmation_msg,
                "debug": debug_text
            })
            with st.chat_message("assistant"):
                st.markdown(confirmation_msg)
                if debug_text:
                    st.caption(debug_text)

            st.session_state.page = 3
            st.rerun()

        # === NO → Keep current type and finish ===
        elif user_response_type == "no":
            no_change_msg = f"Got it — keeping Request Type as **{st.session_state.request_type}**. Finishing up now."
            debug_text = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": no_change_msg,
                "debug": debug_text
            })
            with st.chat_message("assistant"):
                st.markdown(no_change_msg)
                if debug_text:
                    st.caption(debug_text)

            st.session_state.page = 3
            st.rerun()

        # === UNRELATED / QUESTION → Just respond ===
        elif user_response_type in ["unrelated", "question"]:
            debug_text = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": parser_response,
                "debug": debug_text
            })
            with st.chat_message("assistant"):
                st.markdown(parser_response)
                if debug_text:
                    st.caption(debug_text)

        # === MORE CONTEXT → Re-run classification ===
        elif user_response_type == "more context":
            st.session_state.request_description = user_input
            master_llm_result = master_llm(user_input)
            master_flow_result = master_flow_orchestrator(master_llm_result, request_type, user_input)

            st.session_state.master_llm_result = master_llm_result
            st.session_state.master_flow_result = master_flow_result

            new_action = master_flow_result.get("action", "")
            response_text = master_flow_result.get("response_text", "")
            debug_text = show_llm_trace(parse_result, master_llm_result, master_flow_result)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "debug": debug_text
            })
            with st.chat_message("assistant"):
                st.markdown(response_text)
                if debug_text:
                    st.caption(debug_text)

            st.session_state.master_flow_orchestrator_action = new_action
            if new_action == "COMPLETE":
                st.session_state.page = 3
                st.rerun()


# ==========================================================
# 💬 Handle CHATLLM logic — conversational follow-up
# ==========================================================
    elif master_flow_orchestrator_action == "CHATLLM":
        # 1️⃣ Run the chat-response classifier on the user's message
        parse_result = chat_response_parser_llm(user_input)

        user_response_type = parse_result.get("user_response_type", "").lower().strip()
        parser_response = parse_result.get("response", "")

        # 2️⃣ Handle each user response type
        if user_response_type == "yes":
            # ✅ User confirmed they want to change the Request Type
            # Retrieve the most recent classification result (already stored earlier)
            latest_result = st.session_state.master_llm_result.result

            if not latest_result:
                st.warning("⚠️ No LLM result found. Please try again.")
                return

            # 🔹 Update Airtable record
            try:
                update_fields = {"Request Type": latest_result}
                success = update_record(st.session_state.record_id, update_fields)
                if not success:
                    st.warning("⚠️ Could not update Airtable record.")
                else:
                    st.success(f"✅ Airtable updated: Request Type → {latest_result}")
            except Exception as e:
                st.warning(f"Could not update Airtable: {e}")


            # ✅ Confirm change to user and end session
            confirmation_msg = f"✅ Done — I’ve updated the Request Type to **{latest_result}** in your ticket."
            show_llm_trace_debug = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)
            st.session_state.chat_history.append({"role": "assistant", "content": confirmation_msg, "debug": show_llm_trace_debug})
            with st.chat_message("assistant"):
                st.markdown(confirmation_msg)
                show_llm_trace_debug

            # ➡️ Move to Page 3
            st.session_state.page = 3
            st.rerun()


        elif user_response_type == "no":
            # User wants to keep their current Request Type
            no_change_msg = f"Got it — keeping Request Type as **{st.session_state.request_type}**. Finishing up now."
            show_llm_trace_debug = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)
            st.session_state.chat_history.append({"role": "assistant", "content": no_change_msg, "debug": show_llm_trace_debug})
            with st.chat_message("assistant"):
                st.markdown(no_change_msg)
                show_llm_trace_debug

            st.session_state.page = 3
            st.rerun()

        elif user_response_type in ["unrelated", "question"]:
            # 3️⃣ Just display the assistant's response (no flow change)
            show_llm_trace_debug = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)
            st.session_state.chat_history.append({"role": "assistant", "content": parser_response, "debug": show_llm_trace_debug})
            with st.chat_message("assistant"):
                st.markdown(parser_response)
                show_llm_trace_debug
                

        elif user_response_type == "more context":
            # 4️⃣ Rerun master_llm + orchestrator with the new context
            st.session_state.request_description = user_input

            # Run LLM again with updated description
            master_llm_result = master_llm(user_input)

            # Feed into master flow orchestrator
            master_flow_result = master_flow_orchestrator(
                master_llm_result,
                request_type=request_type,
                request_description=user_input
            )

            # ✅ Store both outputs in session for later use
            st.session_state.master_llm_result = master_llm_result
            st.session_state.master_flow_result = master_flow_result

            # Extract updated action + response
            new_action = master_flow_result.get("action", "")
            response_text = master_flow_result.get("response_text", "")
            show_llm_trace_debug = show_llm_trace(parse_result, st.session_state.master_llm_result, st.session_state.master_flow_result)

            # Append and show assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "debug": show_llm_trace_debug})
            with st.chat_message("assistant"):
                st.markdown(response_text)
                show_llm_trace_debug


            # Update current action for next cycle
            st.session_state.master_flow_orchestrator_action = new_action

            # If classification is now complete, move to Page 3
            if new_action == "COMPLETE":
                st.session_state.page = 3
                st.rerun()




# ==============================================================================================================================================================================
# 🧾 PAGE 3 — Ticket Confirmation
# ==============================================================================================================================================================================
def page_3_confirmation():
    """Displays the success confirmation after ticket submission."""

    # 🎈 Balloon animation
    st.balloons()

    # ✅ Airtable updates (only once)
    record_id = st.session_state.get("record_id")
    if record_id and not st.session_state.get("final_updates_done", False):
        # user description update
        update_consolidated_request_description(record_id)
        # full chat history update
        update_chat_history_record(record_id)
        st.session_state.final_updates_done = True

    # 🎟️ Header
    st.title("Ticket Submitted Successfully!")

    # 📋 Retrieve session data
    record_id = st.session_state.get("record_id", "N/A")
    content_title = st.session_state.get("content_title", "(none)")
    request_type = st.session_state.get("request_type", "(none)")
    request_description = st.session_state.get("request_description", "(none)")

    # 🪪 Airtable record link
    AIRTABLE_RECORD_LINK = (
        f"https://airtable.com/appND6VjzjOv4rzUR/pagu1lpeir2Zqi4bw/"
        f"{record_id}?home=paggCoATZHAd9lHfD"
    )

    # 🧾 Ticket Overview Card
    st.markdown(
        f"""
        <div style='
            background-color:#2B2B3A;
            border-radius:16px;
            padding:24px;
            color:white;
            width:75%;
            margin:auto;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            '>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='margin:0;'>Ticket Overview</h3>
                <a href="{AIRTABLE_RECORD_LINK}" target="_blank"
                   style="color:#9DA3FF; text-decoration:none; font-size:18px;"
                   title="Go to Airtable Record">↗</a>
            </div>
            <hr style='border:1px solid #444;'/>
            <p><strong>Content Title:</strong> {content_title}</p>
            <p><strong>Request Type:</strong> {request_type}</p>
            <p><strong>Request Description:</strong> {request_description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 🪶 Footer Text
    st.markdown(
        "<div style='text-align:center; color:gray; margin-top:32px;'>You may now close this tab.</div>",
        unsafe_allow_html=True
    )


# ==========================================================
# 🚦 PAGE CONTROLLER
# ==========================================================
if st.session_state.page == 1:
    page_1_ticket_form()

elif st.session_state.page == 2:
    page_2_llm_chat()

elif st.session_state.page == 3:
    page_3_confirmation()