import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ENV VARS
N8N_WEBHOOK = os.getenv("N8N_CLASSIFY_WEBHOOK")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE_ID")

AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
AIRTABLE_META_URL = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"

# 🔹 Helper: fetch single select options for a field
@st.cache_data
def get_single_select_options(field_name):
    try:
        res = requests.get(
            AIRTABLE_META_URL,
            headers={"Authorization": f"Bearer {AIRTABLE_API_KEY}"},
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        for table in data["tables"]:
            if table["id"] == AIRTABLE_TABLE_ID or table["name"] == AIRTABLE_TABLE_ID:
                for field in table["fields"]:
                    if field["name"] == field_name and field["type"] == "singleSelect":
                        return [c["name"] for c in field["options"]["choices"]]
        return []
    except Exception as e:
        st.error(f"Could not fetch options for {field_name}: {e}")
        return []

# 🔹 UI
st.title("🎟️ IO SST Ticket Submission")

with st.form("ticket_form"):
    # Replace text inputs with Airtable-driven dropdowns
    submitted_team_branch = st.selectbox(
        "Submitted Team Branch",
        get_single_select_options("Submitted Team Branch"),
    )

    requested_by = st.selectbox(
        "Requested By",
        get_single_select_options("Requested By"),
    )

    content_brand = st.text_input("Content Brand")
    content_title = st.text_input("Content Title")

    content_type = st.selectbox("Content Type", ["", "Film", "Series"])

    season_num, episode_num = None, None
    if content_type == "Series":
        season_num = st.number_input("Season #", min_value=1, step=1)
        episode_num = st.number_input("Episode #", min_value=1, step=1)

    request_type = st.selectbox(
        "Request Type",
        get_single_select_options("Request Type"),
    )

    request_text = st.text_area("Request", placeholder="Describe the issue…", height=200)

    submitted = st.form_submit_button("Continue → AI Assist")

# 🔹 Step 2: Call n8n webhook for AI classification
if submitted and request_text:
    with st.spinner("Asking AI assistant..."):
        try:
            res = requests.post(
                N8N_WEBHOOK,
                json={
                    "ticketType": request_type or "I'm Unsure - Described in Request Section",
                    "requestText": request_text,
                },
                timeout=30,
            )
            ai = res.json()
        except Exception as e:
            st.error(f"LLM classification failed: {e}")
            ai = None

    if ai:
        st.subheader("🤖 AI Suggestion")
        st.write(f"**You selected:** {request_type or 'I’m Unsure'}")
        st.write(f"**AI suggests:** {ai.get('inferred_type')} (confidence {ai.get('confidence'):.2f})")
        st.info(f"Reason: {ai.get('reason')}")

        if ai.get("missing_info"):
            st.warning("Missing info needed:")
            for q in ai["missing_info"]:
                st.write(f"- {q}")

        final_type = st.selectbox(
            "Choose the final Ticket Type",
            get_single_select_options("Request Type"),
            index=get_single_select_options("Request Type").index(
                ai.get("inferred_type", "I'm Unsure - Described in Request Section")
            )
            if ai.get("inferred_type") in get_single_select_options("Request Type")
            else 0,
        )

        if st.button("✅ Confirm and Submit to Airtable"):
            # Step 3: Save to Airtable
            fields = {
                "Submitted Team Branch": submitted_team_branch,
                "Requested By": requested_by,
                "Content Brand": content_brand,
                "Content Title": content_title,
                "Content Type": content_type,
                "Request Type": final_type,
                "Request": request_text,
            }
            if content_type == "Series":
                fields["Season #"] = season_num
                fields["Episode #"] = episode_num

            try:
                airtable_res = requests.post(
                    AIRTABLE_URL,
                    headers={
                        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"records": [{"fields": fields}]},
                    timeout=30,
                )
                if airtable_res.status_code == 200:
                    st.success("🎉 Ticket successfully submitted to Airtable!")
                else:
                    st.error(f"Airtable error: {airtable_res.text}")
            except Exception as e:
                st.error(f"Error saving to Airtable: {e}")

