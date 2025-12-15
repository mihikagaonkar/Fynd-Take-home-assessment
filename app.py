import streamlit as st
import pandas as pd
import os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
DATA_FILE = "feedback_data.csv"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="AI Feedback System",
    layout="wide"
)

def init_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "timestamp",
            "user_rating",
            "user_review",
            "ai_response",
            "ai_summary",
            "ai_action"
        ])
        df.to_csv(DATA_FILE, index=False)

init_data()

def ask_groq(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def generate_user_response(review, rating):
    prompt = f"""
You are a polite and empathetic customer support assistant.

User Rating: {rating} stars
User Review: "{review}"

Write a friendly, professional response to the user.
"""
    return ask_groq(prompt)

def generate_summary(review):
    prompt = f"""
Summarize the following customer review in ONE short sentence.

Review:
"{review}"
"""
    return ask_groq(prompt)

def generate_action(review, rating):
    prompt = f"""
Based on the customer review and rating below,
suggest ONE clear recommended action for the business team.

Rating: {rating}
Review: "{review}"
"""
    return ask_groq(prompt)

st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Choose Dashboard",
    ["User Dashboard", "Admin Dashboard"]
)

if mode == "User Dashboard":
    st.title("📝 User Feedback")

    rating = st.selectbox(
        "Select Rating",
        [1, 2, 3, 4, 5]
    )

    review = st.text_area(
        "Write your review",
        height=150
    )

    if st.button("Submit Feedback"):
        if review.strip() == "":
            st.warning("Please enter a review before submitting.")
        else:
            with st.spinner("Generating AI response..."):
                ai_response = generate_user_response(review, rating)
                ai_summary = generate_summary(review)
                ai_action = generate_action(review, rating)

            new_row = {
                "timestamp": datetime.utcnow(),
                "user_rating": rating,
                "user_review": review,
                "ai_response": ai_response,
                "ai_summary": ai_summary,
                "ai_action": ai_action
            }

            df = pd.read_csv(DATA_FILE)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)

            st.success("Feedback submitted successfully!")

            st.subheader("🤖 AI Response")
            st.write(ai_response)

if mode == "Admin Dashboard":
    st.title("📊 Admin Dashboard")

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        st.info("No feedback submissions yet.")
    else:
        col1, col2 = st.columns(2)

        col1.metric("Total Submissions", len(df))
        col2.metric(
            "Average Rating",
            round(df["user_rating"].mean(), 2)
        )

        st.subheader("📋 All Feedback (Latest First)")
        st.dataframe(
            df.iloc[::-1],
            width='stretch'
        )

        st.subheader("📈 Rating Distribution")
        st.bar_chart(
            df["user_rating"].value_counts().sort_index()
        )
