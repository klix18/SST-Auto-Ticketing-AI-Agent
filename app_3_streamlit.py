import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# LangChain / LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

from app_3_llm import rag_classify, summarize_type_with_rag, llm_chat_reply, get_llm_json, get_pc_clients, rag_retrieve_top3

# ==========================================================
# 🔐 ENV
# ==========================================================
load_dotenv()
OPENAI_API_KEY               = os.getenv("OPENAI_API_KEY")  # must be set
AIRTABLE_API_KEY             = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID             = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_MAIN_TABLE_ID       = os.getenv("AIRTABLE_MAIN_TABLE_ID")
AIRTABLE_REQUESTORS_TABLE_ID = os.getenv("AIRTABLE_REQUESTORS_TABLE_ID")
AIRTABLE_WATCHLIST_TABLE_ID  = os.getenv("AIRTABLE_WATCHLIST_TABLE_ID")

# Pinecone (new SDK v5)
PINECONE_API_KEY             = os.getenv("PINECONE_API_KEY")
PINECONE_HOST                = "https://sst-master-db-xxzkqyr.svc.aped-4627-b74a.pinecone.io"
EMBED_MODEL                  = "text-embedding-3-small"  # dims=512

HEADERS = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
AIRTABLE_BASE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
AIRTABLE_META_URL = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"

# ---- RAG configuration ----
EMBED_DIM = 512  # your Pinecone index dimension



# ==========================================================
# ✅ REQUIRED-FIELD SWITCHES (all False by default)
# ==========================================================
MANDATORY = {
    "Submitted Team Branch": False,
    "Requested By": False,
    "Watch List": False,
    "Content Brand": False,
    "Content Title": False,
    "Content Type": False,
    "Season #": False,       # only when Content Type == "Series"
    "Episode #": False,      # only when Content Type == "Series"
    "Request Type": True,
    "Request Description": True,  # free text
}
# ==========================================================
# 🔧 CATEGORIES
# ==========================================================
CATEGORIES = [
    "Make New Package",
    "Publish Artwork to Platform",
    "Change Existing Image Assets",
    "Add Missing Image Assets",
]

# ==========================================================
# 🔧 Helpers
# ==========================================================
@st.cache_data
def get_select_options(field_name: str):
    """Return option names for MAIN table field (singleSelect or multipleSelects)."""
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
    """Return list of {'id','name'} from a linked table."""
    try:
        r = requests.get(f"{AIRTABLE_BASE_URL}/{table_id}", headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return [
            {"id": rec["id"], "name": rec.get("fields", {}).get(display_field, f"Unnamed ({rec['id']})")}
            for rec in data.get("records", [])
        ]
    except Exception as e:
        st.error(f"Error fetching linked records from {table_id}: {e}")
        return []

def fetch_main_record(record_id: str):
    try:
        r = requests.get(f"{AIRTABLE_BASE_URL}/{AIRTABLE_MAIN_TABLE_ID}/{record_id}", headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()  # {'id':..., 'fields': {...}}
    except Exception as e:
        st.error(f"Error reading record {record_id}: {e}")
        return None

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

def update_main_record(record_id: str, fields: dict):
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
        st.error(f"Error updating record {record_id}: {e}")
        return False

# --- extractors from Airtable record ---
def get_single_value(fields: dict, name: str) -> str:
    v = fields.get(name)
    if v is None:
        return ""
    if isinstance(v, dict) and "name" in v:  # singleSelect
        return v["name"]
    if isinstance(v, str):
        return v
    return ""

def get_multi_select_values(fields: dict, name: str):
    v = fields.get(name) or []
    out = []
    for item in v:
        if isinstance(item, dict) and "name" in item:
            out.append(item["name"])
        elif isinstance(item, str):
            out.append(item)
    return out

def get_linked_ids(fields: dict, name: str):
    v = fields.get(name) or []
    return [rid for rid in v if isinstance(rid, str)]

def is_missing(label: str, value) -> bool:
    """Return True if missing AND label is mandatory; show warning."""
    if not MANDATORY.get(label, False):
        return False
    if value is None:
        st.warning(f"⚠️ '{label}' is required.")
        return True
    if isinstance(value, str) and value.strip() == "":
        st.warning(f"⚠️ '{label}' is required.")
        return True
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        st.warning(f"⚠️ '{label}' is required.")
        return True
    return False



# ==========================================================
# 🔧 UI Helpers
# ==========================================================

# --- Mandatory UI helpers ---
def _is_present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, tuple, set)):
        return len(v) > 0
    return True

def render_label(label: str):
    """Render a label with a red * if the field is mandatory."""
    star = " <span style='color:#e00'>&nbsp;*</span>" if MANDATORY.get(label, False) else ""
    st.markdown(
        f"<div style='font-weight:600; margin:2px 0 4px 0;'>{label}{star}</div>",
        unsafe_allow_html=True
    )

def labeled_selectbox(label, options, key, **kwargs):
    render_label(label)
    return st.selectbox("", options, key=key, label_visibility="collapsed", **kwargs)

def labeled_multiselect(label, options, key, **kwargs):
    render_label(label)
    return st.multiselect("", options, key=key, label_visibility="collapsed", **kwargs)

def labeled_text_input(label, key, **kwargs):
    render_label(label)
    return st.text_input("", key=key, label_visibility="collapsed", **kwargs)

def labeled_text_area(label, key, **kwargs):
    render_label(label)
    return st.text_area("", key=key, label_visibility="collapsed", **kwargs)



# ==========================================================
# 🧭 Session defaults
# ==========================================================
st.session_state.setdefault("page", 1)
st.session_state.setdefault("record_id", None)
st.session_state.setdefault("hydrated_p1", False)

# RAG state & chat state
st.session_state.setdefault("classification", None)   # dict with 4 keys
st.session_state.setdefault("chat_history", [])       # list of {role, content}
st.session_state.setdefault("chat_enabled", False)    # submit availability rule
st.session_state.setdefault("final_request_type", None)
st.session_state.setdefault("request_type", "")
st.session_state.setdefault("request_description", "")

# Page 3 countdown state
st.session_state.setdefault("countdown_started", False)
st.session_state.setdefault("countdown_target", 0.0)

# ==========================================================
# 🧾 Page 1 — Ticket Details (then Next → chat)
# ==========================================================
if st.session_state.page == 1:
    st.title("🎟️ SST Ticket • Step 1 of 2 — Ticket Details")

    # Hydrate from Airtable if editing an existing record (first render only)
    if st.session_state.record_id and not st.session_state.hydrated_p1:
        rec = fetch_main_record(st.session_state.record_id)
        if rec:
            f = rec.get("fields", {})
            st.session_state.setdefault("submitted_team_branch", get_single_value(f, "Submitted Team Branch"))
            st.session_state.setdefault("requested_by_name", "")
            st.session_state.setdefault("requested_by_id", None)
            st.session_state.setdefault("watchlist_ids", get_linked_ids(f, "Watch List"))
            st.session_state.setdefault("watchlist_names", [])
            st.session_state.setdefault("content_brand", get_single_value(f, "Content Brand"))
            st.session_state.setdefault("content_title", get_single_value(f, "Content Title"))
            st.session_state.setdefault("content_type", get_single_value(f, "Content Type"))
            st.session_state.setdefault("season_num", get_single_value(f, "Season #"))
            st.session_state.setdefault("episode_nums", get_multi_select_values(f, "Episode #"))
            st.session_state.setdefault("request_type", get_single_value(f, "Request Type"))
            st.session_state.setdefault("request_description", get_single_value(f, "Request Description"))
        st.session_state.hydrated_p1 = True

    # Options
    branch_opts  = get_select_options("Submitted Team Branch")
    brand_opts   = get_select_options("Content Brand")
    type_opts    = get_select_options("Content Type")
    season_opts  = get_select_options("Season #")
    episode_opts = get_select_options("Episode #")   # supports multipleSelects
    reqtype_opts = get_select_options("Request Type")

    reqs         = get_linked_record_options(AIRTABLE_REQUESTORS_TABLE_ID)
    req_names    = [r["name"] for r in reqs]
    watch        = get_linked_record_options(AIRTABLE_WATCHLIST_TABLE_ID)
    watch_names  = [w["name"] for w in watch]

    # Map stored watchlist IDs -> names (first load only)
    if st.session_state.record_id and watch_names and st.session_state.get("watchlist_ids") and not st.session_state.get("watchlist_names"):
        id_to_name = {w["id"]: w["name"] for w in watch}
        st.session_state.watchlist_names = [id_to_name[i] for i in st.session_state.watchlist_ids if i in id_to_name]

    # Resolve requested_by_name from existing linked id
    if st.session_state.record_id and req_names:
        if st.session_state.get("requested_by_id") and not st.session_state.get("requested_by_name"):
            id_to_name = {r["id"]: r["name"] for r in reqs}
            st.session_state.requested_by_name = id_to_name.get(st.session_state.requested_by_id, "")

    # Widgets
    st.selectbox("Submitted Team Branch", branch_opts or ["(No options)"], key="submitted_team_branch")
    st.selectbox("Requested By", req_names or ["(No options)"], key="requested_by_name")
    st.multiselect("Watch List", watch_names or ["(No options)"], key="watchlist_names")

    st.selectbox("Content Brand", brand_opts or ["(No options)"], key="content_brand")
    st.text_input("Content Title", key="content_title")
    st.selectbox("Content Type", type_opts or ["(No options)"], key="content_type")

    if st.session_state.content_type == "Series":
        st.subheader("📺 Series Details")
        st.selectbox("Season #", season_opts or ["(No options)"], key="season_num")
        st.multiselect("Episode #", episode_opts or ["(No options)"], key="episode_nums")
    else:
        st.session_state.season_num = ""
        st.session_state.episode_nums = []

    labeled_selectbox("Request Type", reqtype_opts or ["(No options)"], key="request_type")
    labeled_text_area("Request Description", key="request_description", height=200,
                  placeholder="Please describe your request as detailed as possible and avoid ambiguous phrasing. Thank you!")

    # Keep derived IDs in sync for writes
    if req_names:
        sel = next((r for r in reqs if r["name"] == st.session_state.requested_by_name), None)
        st.session_state.requested_by_id = sel["id"] if sel else None
    if watch_names:
        selset = set(st.session_state.watchlist_names or [])
        st.session_state.watchlist_ids = [w["id"] for w in watch if w["name"] in selset]

    # --- NEXT: disabled if mandatory not selected
        # Compute disabled state from MANDATORY + current values
    required_now = [
        ("Submitted Team Branch", st.session_state.submitted_team_branch),
        ("Requested By",         st.session_state.requested_by_name),
        ("Watch List",           st.session_state.watchlist_names),
        ("Content Brand",        st.session_state.content_brand),
        ("Content Title",        st.session_state.content_title),
        ("Content Type",         st.session_state.content_type),
        ("Request Type",         st.session_state.request_type),
        ("Request Description",  st.session_state.request_description),
    ]
    if st.session_state.content_type == "Series":
        required_now += [("Season #", st.session_state.season_num),
                        ("Episode #", st.session_state.episode_nums)]

    def _empty(v):
        if v is None: return True
        if isinstance(v, str): return v.strip() == ""
        if isinstance(v, (list, tuple, set)): return len(v) == 0
        return False

    disable_next = any(
        MANDATORY.get(name, False) and _empty(val)
        for name, val in required_now
    )


    # --- NEXT: validate → create/update → RAG classify → navigate (inline) ---
    if st.button("Next →", type="primary", disabled=disable_next):

        # validation
        missing = any([
            is_missing("Submitted Team Branch", st.session_state.submitted_team_branch),
            is_missing("Requested By",        st.session_state.requested_by_name),
            is_missing("Watch List",          st.session_state.watchlist_names),
            is_missing("Content Brand",       st.session_state.content_brand),
            is_missing("Content Title",       st.session_state.content_title),
            is_missing("Content Type",        st.session_state.content_type),
            is_missing("Request Type",        st.session_state.request_type),
            is_missing("Request Description", st.session_state.request_description),
        ])
        if st.session_state.content_type == "Series":
            missing = missing or any([
                is_missing("Season #",  st.session_state.season_num),
                is_missing("Episode #", st.session_state.episode_nums),
            ])
        if missing:
            st.stop()

        # write to Airtable (create or update)
        fields = {
            "Submitted Team Branch": st.session_state.submitted_team_branch or None,
            "Requested By": [st.session_state.requested_by_id] if st.session_state.requested_by_id else [],
            "Watch List": st.session_state.watchlist_ids or [],
            "Content Brand": st.session_state.content_brand or None,
            "Content Title": st.session_state.content_title or None,
            "Content Type": st.session_state.content_type or None,
            "Request Type": st.session_state.request_type or None,
            "Request Description": st.session_state.request_description or None,
        }
        if st.session_state.content_type == "Series":
            fields["Season #"]  = st.session_state.season_num or None
            fields["Episode #"] = st.session_state.episode_nums or []
        else:
            fields["Season #"]  = None
            fields["Episode #"] = []

        if st.session_state.record_id is None:
            rid = create_main_record(fields)
            if not rid:
                st.stop()
            st.session_state.record_id = rid
        else:
            ok = update_main_record(st.session_state.record_id, fields)
            if not ok:
                st.stop()

        # ========= RAG CLASSIFY (IMMEDIATE on NEXT) =========
        classification = rag_classify(
            request_type=st.session_state.request_type,
            request_description=st.session_state.request_description,
        )
        st.session_state.classification = classification
        st.session_state.final_request_type = st.session_state.request_type

        suggested = classification.get("result", "").strip()
        rq_type = st.session_state.request_type.strip()

        # If model agrees with user's chosen Request Type → skip chat
        if suggested and rq_type and suggested.lower() == rq_type.lower():
            history_payload = {
                "classification": classification,
                "chat_history": [],  # no chat needed
            }
            update_main_record(st.session_state.record_id, {
                "Chatbot Chat History": json.dumps(history_payload, ensure_ascii=False),
                "Request Type": rq_type,
            })
            st.session_state.page = 3
            st.session_state.countdown_started = False
            st.rerun()
        else:
            # Initialize chat container for Page 2
            st.session_state.chat_history = []
            st.session_state.chat_enabled = False
            st.session_state.page = 2
            st.rerun()


# ==========================================================
# 🗨️ Page 2 — Chat or success
# ==========================================================
elif st.session_state.page == 2:
    st.title("💬 SST Ticket • Step 2 of 2 — Chat")

    if not st.session_state.record_id or not st.session_state.classification:
        st.warning("No ticket or classification yet. Please complete Step 1.")
    else:
        cls = st.session_state.classification

        # Safely read; if missing, hydrate from Airtable
        rq_type = st.session_state.get("request_type", "")
        rq_desc = st.session_state.get("request_description", "")

        if (not rq_type or not rq_desc) and st.session_state.get("record_id"):
            rec = fetch_main_record(st.session_state.record_id)
            if rec:
                f = rec.get("fields", {})
                if not rq_type:
                    st.session_state.request_type = get_single_value(f, "Request Type")
                    rq_type = st.session_state.request_type
                if not rq_desc:
                    st.session_state.request_description = get_single_value(f, "Request Description")
                    rq_desc = st.session_state.request_description

        suggested = cls.get("result", "").strip()
        ctx = cls.get("context_used", "")

        # If classifier could not decide strictly from RAG
        if suggested == "Insufficient Context":
            st.warning("I don't have enough RAG context to classify this. Please add more detail in the chat below.")

        # CASE A: Model agrees with user selection → show success and Done
        if suggested and rq_type and suggested.lower() == rq_type.lower():
            st.success("✅ It looks like your request went through correctly.")
            st.caption(f"Model agreed with '{rq_type}'. Confidence: {cls.get('confidence', 0)}")
            if st.button("Done"):
                history_payload = {
                    "classification": cls,
                    "chat_history": st.session_state.chat_history,
                }
                update_main_record(st.session_state.record_id, {
                    "Chatbot Chat History": json.dumps(history_payload, ensure_ascii=False)
                })
                # → Go to Page 3 (auto-close countdown)
                st.session_state.page = 3
                st.session_state.countdown_started = False
                st.rerun()

        else:
            # CASE B: Model disagrees → open chat UI
            # Seed initial assistant message once with real summaries from RAG
            if not st.session_state.chat_history:
                current_summary = summarize_type_with_rag(rq_type, ctx)
                suggested_summary = summarize_type_with_rag(suggested, ctx)

                first_msg = (
                    f"It looks like you chose '{rq_type}' for this ticket. "
                    f"However, based on your Request Description '{rq_desc}', "
                    f"I think '{suggested}' is the request type you're actually looking for.\n\n"
                    f"{current_summary}\n\n"
                    f"{suggested_summary}\n\n"
                    f"Would you like me to change the Request Type to '{suggested}' or continue with your previous choice?"
                )
                st.session_state.chat_history.append({"role": "assistant", "content": first_msg})

            # Render chat
                        # Render chat with metadata
            for i, m in enumerate(st.session_state.chat_history):
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

                    # After first assistant message, show classification metadata
                    if i == 0 and m["role"] == "assistant":
                        cls = st.session_state.classification or {}
                        conf = cls.get("confidence", 0)
                        chunks = cls.get("chunks_used", [])
                        chunk_str = ", ".join(chunks) if chunks else "N/A"
                        st.caption(f"**Confidence:** {conf}% · **Chunks Used:** {chunk_str}")


            user_input = st.chat_input("Type your reply…")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_enabled = True  # enable submit

                lower = user_input.lower()

                wants_no_change = any(k in lower for k in [
                    "no", "dont change", "don't change", "do not change", "continue",
                    "keep", "keep it", "leave it", "no change", "stick with", "stay",
                    "keep previous", "use previous", "keep original", "keep current"
                ])
                wants_change = any(k in lower for k in [
                    "change", "switch", "update", "yes change", "use your",
                    "use suggested", "yes", "ok", "sure", "sounds good"
                ])
                wants_more  = any(k in lower for k in ["more", "explain", "why", "details"])

                if wants_no_change:
                    # Keep user's original type; proceed to Page 3
                    reply = f"Got it — keeping Request Type as **{rq_type}**. Finishing up now."
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                    history_payload = {
                        "classification": cls,
                        "chat_history": st.session_state.chat_history,
                    }
                    update_main_record(st.session_state.record_id, {
                        "Chatbot Chat History": json.dumps(history_payload, ensure_ascii=False),
                        "Request Type": rq_type,  # keep original
                    })
                    st.session_state.final_request_type = rq_type
                    st.session_state.page = 3
                    st.session_state.countdown_started = False
                    st.rerun()

                elif wants_change and suggested and suggested != "Insufficient Context":
                    # update Request Type in Airtable to suggested
                    if update_main_record(st.session_state.record_id, {"Request Type": suggested}):
                        st.session_state.final_request_type = suggested
                        reply = f"Done — I changed Request Type to **{suggested}** in the ticket. Anything else you’d like to review?"
                    else:
                        reply = "I tried to change the Request Type but ran into an issue. Would you like me to try again?"
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                elif wants_more:
                    reply = llm_chat_reply(user_input, ctx, rq_type, rq_desc, suggested)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                else:
                    # ask for clarification or give a helpful context-based answer
                    reply = llm_chat_reply(user_input, ctx, rq_type, rq_desc, suggested)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

            # Final submit (enabled once user chatted at least once)
            disabled = not st.session_state.chat_enabled
            if st.button("Submit", disabled=disabled):
                history_payload = {
                    "classification": cls,                       # 4-key JSON
                    "chat_history": st.session_state.chat_history,
                }
                update_main_record(st.session_state.record_id, {
                    "Chatbot Chat History": json.dumps(history_payload, ensure_ascii=False),
                    "Request Type": st.session_state.final_request_type or rq_type,
                })
                # → Go to Page 3 (auto-close countdown)
                st.session_state.page = 3
                st.session_state.countdown_started = False
                st.rerun()

# ==========================================================
# ✅ Page 3 — Auto-close countdown (visible text + JS close)
# ==========================================================
elif st.session_state.page == 3:
    # 🎉 Success header + one-time balloons
    st.title("✅ Ticket Submitted Successfully!")
    if not st.session_state.get("celebrated"):
        st.balloons()
        st.session_state.celebrated = True

    # --- Gather summary values (prefer session; fallback to Airtable) ---
    content_title = st.session_state.get("content_title", "")
    request_type = st.session_state.get("request_type", "")
    request_description = st.session_state.get("request_description", "")

    if (not content_title or not request_type or not request_description) and st.session_state.get("record_id"):
        rec = fetch_main_record(st.session_state.record_id)
        if rec:
            f = rec.get("fields", {})
            content_title = content_title or get_single_value(f, "Content Title")
            request_type = request_type or get_single_value(f, "Request Type")
            request_description = request_description or get_single_value(f, "Request Description")

    # Basic sanitization for display
    content_title_disp = content_title or "(none)"
    request_type_disp = request_type or "(none)"
    # Escape double-quotes in description so our quoted display looks clean
    request_desc_disp = (request_description or "(none)").replace('"', '\\"')

    # --- Centered summary card ---
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("### You submitted a ticket for:")
        st.markdown(
            f"""
**Content Title:** '{content_title_disp}'  
**Request Type:** "{request_type_disp}"  
**Request Description:** "{request_desc_disp}"
            """.strip()
        )

    # --- Footer note ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: gray;'>You may now close this tab</div>",
        unsafe_allow_html=True,
    )



