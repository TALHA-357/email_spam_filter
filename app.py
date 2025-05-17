import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  # Download the punkt tokenizer
# Custom CSS for styling
st.markdown("""
<style>
    .title {
        color: purple;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        background-color: lightgreen;
        border-radius: 10px;
    }
    .subheader {
        color: white;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .spam-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .ham-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stTextArea textarea {
        min-height: 200px;
    }
    .stButton button {
        background-color: #4a4a4a;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

class NaiveBayesClassifier:
    @classmethod
    def load_model(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls()
        classifier.df2 = data['df2']
        classifier.prob_ham = data['prob_ham']
        classifier.prob_spam = data['prob_spam']
        classifier.words = data['words']
        return classifier
    
    def predict(self, email):
        email = email.lower()
        test_list = word_tokenize(email)
        
        result1 = 1
        for test_word in test_list:
            for i in range(len(self.df2)):
                myrow = self.df2.iloc[i]
                if myrow['Words'] == test_word:
                    result1 = result1 * myrow['Spam Count Prob']
        
        result1 = result1 * self.prob_spam
        
        result2 = 1
        for test_word in test_list:
            for i in range(len(self.df2)):
                myrow = self.df2.iloc[i]
                if myrow['Words'] == test_word:
                    result2 = result2 * myrow['Ham Count Prob']
        
        result2 = result2 * self.prob_ham
        
        final = result1 + result2
        
        spam_prob = (result1 / final) * 100 if final != 0 else 0
        ham_prob = (result2 / final) * 100 if final != 0 else 0
        
        return spam_prob, ham_prob

# Load the model
@st.cache_resource
def load_model():
    return NaiveBayesClassifier.load_model('naive_bayes_model.pkl')

classifier = load_model()

# App layout
st.markdown('<h1 class="title">Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter an email to check if it\'s spam or ham</p>', unsafe_allow_html=True)

# Email input
email_text = st.text_area("Email Text:", placeholder="Paste your email here...", height=250)

# Classify button
if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter an email to classify")
    else:
        with st.spinner("Analyzing the email..."):
            spam_prob, ham_prob = classifier.predict(email_text)
        
        # Display results
        if spam_prob > ham_prob:
            result_class = "spam-result"
            result_text = "SPAM"
            icon = "⚠️"
            color = "#f44336"
        else:
            result_class = "ham-result"
            result_text = "HAM (Not Spam)"
            icon = "✅"
            color = "#4caf50"
        
        st.markdown(f"""
        <div class="result-box {result_class}">
            <h2 style="color: {color}; text-align: center;">{icon} This email is {result_text}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(['Spam', 'Ham'], [spam_prob, ham_prob], 
                   color=['#f44336', '#4caf50'])
            ax.set_ylim(0, 100)
            ax.set_ylabel('Probability (%)')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Confidence Meter")
            confidence = abs(spam_prob - ham_prob)
            
            if confidence > 70:
                confidence_text = "High confidence"
                emoji = "🔍"
                bar_color = "#4caf50" if result_text == "HAM (Not Spam)" else "#f44336"
            elif confidence > 40:
                confidence_text = "Medium confidence"
                emoji = "🤔"
                bar_color = "#ff9800"
            else:
                confidence_text = "Low confidence"
                emoji = "🧐"
                bar_color = "#9e9e9e"
            
            st.markdown(f"""
            <div style="margin-top: 20px;">
                <p style="font-size: 1.2em;">{emoji} {confidence_text}</p>
                <div style="height: 10px; background: #e0e0e0; border-radius: 5px;">
                    <div style="height: 10px; width: {confidence}%; background: {bar_color}; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px;">{confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Word analysis
        st.markdown("### Word Analysis")
        email_words = [word.lower() for word in word_tokenize(email_text)]
        spam_words = []
        ham_words = []
        
        for word in set(email_words):
            for i in range(len(classifier.df2)):
                myrow = classifier.df2.iloc[i]
                if myrow['Words'] == word:
                    if myrow['Spam Count Prob'] > myrow['Ham Count Prob']:
                        spam_words.append(word)
                    else:
                        ham_words.append(word)
        
        if spam_words:
            st.markdown("🔴 **Spam indicators found:** " + ", ".join(spam_words[:10]))
        if ham_words:
            st.markdown("🟢 **Ham indicators found:** " + ", ".join(ham_words[:10]))
        
        # Raw probabilities
        with st.expander("Show detailed probabilities"):
            st.write(f"Spam probability: {spam_prob:.2f}%")
            st.write(f"Ham probability: {ham_prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d;">
    <p>Naive Bayes Classifier | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)