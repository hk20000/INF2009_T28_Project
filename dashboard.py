import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import httpx
import time
import json
import csv
import os

# Optional (for auto refresh)
from streamlit_autorefresh import st_autorefresh


# -------------------------------------------
# Hardcode your Groq API Key here:
# -------------------------------------------
HARDCODED_API_KEY = "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"

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Real-Time Engagement Dashboard", 
    page_icon=":bar_chart:", 
    layout="wide"
)
st.title("Classroom Engagement Dashboard :bar_chart:")
st.markdown(
    "Monitor real-time facial expressions and transcriptions in the classroom "
    "to gauge overall engagement levels at a glance."
)

# -------------------------------
# Database Connection
# -------------------------------
conn = sqlite3.connect('engagement.db', check_same_thread=False)
c = conn.cursor()

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_data():
    c.execute('SELECT * FROM engagement ORDER BY timestamp DESC LIMIT 20')
    return c.fetchall()

def calculate_engagement(data):
    if not data:
        return 0  # No data, default engagement = 0

    total = len(data)
    engaged_count = 0

    # Define weightage for different expressions
    for row in data:
        if row[2] == 'engaged':
            engaged_count += 1  # Full engagement
        elif row[2] == 'talking':
            engaged_count += 0.5  # Partial engagement
        # 'sleeping' is treated as no engagement, so no need to add to engaged_count
    
    # Calculate the engagement percentage based on weighted count
    return round((engaged_count / total) * 100, 2)


def draw_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Engagement Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {
                'color': "green" if value > 60 else "orange" if value > 30 else "red"
            },
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "green"},
            ],
        }
    ))
    return fig

def send_data_to_groq_chat(data):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HARDCODED_API_KEY}",
        "Content-Type": "application/json"
    }
    
    with httpx.Client(verify=False) as client:  # Disable SSL verification
        data_json = json.dumps(data)
        
        payload = {
            "model": "llama-3.3-70b-versatile",  # Example model
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data_json}
            ]
        }
        
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error from Groq API: {response.status_code}")
                return None
        except httpx.RequestError as e:
            st.error(f"An error occurred while sending data to Groq API: {e}")
            return None

def generate_engagement_summary(data):
    if not data:
        return "No data available for analysis."
    
    engaged_count = sum(1 for row in data if row[2] == 'engaged')
    total_count = len(data)
    engagement_percentage = (engaged_count / total_count) * 100 if total_count else 0

    if engagement_percentage > 75:
        summary = f"Excellent engagement! {engaged_count} out of {total_count} participants are engaged ({engagement_percentage:.2f}%)."
    elif engagement_percentage > 50:
        summary = f"Good engagement. {engaged_count} out of {total_count} participants are engaged ({engagement_percentage:.2f}%)."
    else:
        summary = f"Low engagement. Only {engaged_count} out of {total_count} participants are engaged ({engagement_percentage:.2f}%)."
    
    return summary

def save_llm_response_to_csv(speaker, llm_response, csv_file="speaker_llm_responses.csv"):
    """Append the (speaker, llm_response) to a local CSV file."""
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Speaker", "LLM_Response"])
        writer.writerow([speaker, llm_response])

# We use Streamlit session state to avoid multiple triggers for the same speaker
if "last_processed_speaker" not in st.session_state:
    st.session_state["last_processed_speaker"] = None

# -------------------------------
# Auto-check function
# -------------------------------
def auto_check_and_run_llm():
    """Checks if the latest 20 rows are from a single speaker,
       and if so, sends them to the LLM (if it’s a new speaker)."""
    data = fetch_data()
    # You can adjust this threshold from 10 back to 20 if you wish
    if len(data) < 10:
        return  # Not enough rows to do the check

    df = pd.DataFrame(data, columns=["Speaker", "Timestamp", "Expression", "Transcription"])
    unique_speakers = df["Speaker"].unique()

    # If all rows are from a single speaker
    if len(unique_speakers) == 1:
        current_speaker = unique_speakers[0]
        # Only proceed if it's a *new* speaker we haven't processed yet
        if current_speaker != st.session_state["last_processed_speaker"]:
            # Send to LLM
            insights = send_data_to_groq_chat(df.to_dict(orient='records'))
            if insights:
                try:
                    summary_text = insights["choices"][0]["message"]["content"]
                    # Save to CSV
                    save_llm_response_to_csv(current_speaker, summary_text)
                    st.session_state["last_processed_speaker"] = current_speaker
                except (KeyError, IndexError, TypeError):
                    # If structure is unexpected, do minimal handling
                    pass

# -------------------------------
# Main Display Logic
# -------------------------------
def display_engagement_data():
    data = fetch_data()
    engagement_score = calculate_engagement(data)

    if data:
        df = pd.DataFrame(data, columns=["Speaker", "Timestamp", "Expression", "Transcription"])

        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.subheader("Latest Engagement Data")
            st.dataframe(df, use_container_width=True)

        with col2:
            st.subheader("Engagement Gauge")
            st.plotly_chart(draw_gauge(engagement_score), use_container_width=True)

        ai_summary = generate_engagement_summary(data)
        st.info(ai_summary)

        # Provide a download button for the CSV if it exists
        if os.path.exists("speaker_llm_responses.csv"):
            with open("speaker_llm_responses.csv", "r", encoding="utf-8") as f:
                csv_data = f.read()
            st.download_button(
                label="Download LLM Responses CSV",
                data=csv_data,
                file_name="speaker_llm_responses.csv",
                mime="text/csv"
            )

            # ---- NEW PART: SHOW THE CSV CONTENT ----
            st.subheader("Content of the CSV File")
            csv_df = pd.read_csv("speaker_llm_responses.csv")
            st.dataframe(csv_df, use_container_width=True)

    else:
        st.warning("No data available in the database.")

# -------------------------------
# Auto refresh every X seconds
# -------------------------------
# This will cause the script to re-run automatically every 10 seconds
st_autorefresh(interval=10_000, limit=100_000, key="auto_refresh")

# Run the auto-check logic on every script re-run
auto_check_and_run_llm()

# -------------------------------
# Sidebar for Manual Refresh
# -------------------------------
st.sidebar.header("Controls")
if st.sidebar.button("End"):
    display_engagement_data()
else:
    # Optionally display data by default if you want
    display_engagement_data()
    #st.info("The page also auto-checks for a single speaker every 10 seconds.")
