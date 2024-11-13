import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st

# Download NLTK stop words (only needed once)
nltk.download('stopwords')

# Define the path to your CSV file
CSV_FILE_PATH ="assistance_responses_cvs_new.csv"

# Custom CSS for Modern Flat Design
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;  /* Light blue background */
        font-family: 'Arial', sans-serif;
    }
    .header {
        background-color: #0078d4;  /* Blue header */
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .input-area {
        margin: 20px 0;
    }
    .button {
        background-color: #0078d4;  /* Blue button */
        color: white; 
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .button:hover {
        background-color: #005a9e;  /* Darker blue on hover */
    }
    .response {
        background-color: #e1f5fe;  /* Light blue response box */
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.1);
    }
    .centered {
        text-align: center;  /* Center text */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown("<div class='header'><h1>🤖 Help Me Chatbot</h1></div>", unsafe_allow_html=True)
st.markdown("<h2 class='centered'>Welcome to Our Chatbot!</h2>", unsafe_allow_html=True)

st.markdown("<div class='centered'>I'm here to help you with your queries. Please enter your question below:</div>",
            unsafe_allow_html=True)


def handle_error(exception):
    """Display an error message in Streamlit."""
    st.error(f"An error occurred: {exception}")


# Attempt to read the CSV file with specified encoding
try:
    data = pd.read_csv(CSV_FILE_PATH, encoding='latin1')  # Change encoding if needed

    # Check if the required columns exist
    if all(col in data.columns for col in ['Category', 'Question', 'Response']):
        questions = data['Question'].tolist()
        categories = data['Category'].tolist()
        responses = dict(zip(data['Category'], data['Response']))  # Create a dictionary for responses

        # Preprocessing with TF-IDF and using a Multinomial Naive Bayes classifier
        stop_words = stopwords.words('english')
        model = make_pipeline(
            TfidfVectorizer(stop_words=stop_words),
            MultinomialNB()
        )

        # Split into training and testing sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            questions, categories, test_size=0.2, random_state=42, stratify=categories
        )

        # Train the model
        model.fit(X_train, y_train)


        def classify_message(message):
            """Predict the category of the input message."""
            return model.predict([message])[0]


        # User input for classification
        user_input = st.text_input("Your Question:", "", key="input", placeholder="Type your question here...")

        if st.button("Get Help", key="help_button"):
            if user_input:
                category = classify_message(user_input)
                st.markdown(f"<div class='response'>Chatbot: Your query is related to **'{category}'**.</div>",
                            unsafe_allow_html=True)

                # Provide assistance based on the identified category
                if category in responses:
                    st.markdown(f"<div class='response'>Assistance: {responses[category]}</div>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='response'>Assistance: I'm not sure how to help with that. Please contact customer support.</div>",
                        unsafe_allow_html=True)
            else:
                st.warning("Please enter a query before clicking the button.")
    else:
        st.error("The CSV file must contain 'Category', 'Question', and 'Response' columns.")

except FileNotFoundError:
    st.error(f"Error: The file '{CSV_FILE_PATH}' was not found. Please check the path.")
except Exception as e:
    handle_error(e)
