import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pickle
import os

class NaiveBayesClassifier:
    def __init__(self):
        self.df2 = None
        self.prob_ham = None
        self.prob_spam = None
        self.words = None
    
    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        df['email'] = df['email'].str.lower()
        
        words = np.array([])
        for item in df['email']:
            token = word_tokenize(item)
            for i in token:
                if i not in words:
                    words = np.append(words, i)
        
        word_list = np.array([])
        spam_list = np.array([])
        ham_list = np.array([])
        spam_count = 0
        ham_count = 0
        
        for word in words:
            for i in range(len(df)):
                row = df.iloc[i]
                if((word in row['email']) and (row['label'] == 'spam')):
                    spam_count += 1
                elif((word in row['email']) and (row['label'] == 'ham')):
                    ham_count += 1
            word_list = np.append(word_list, word)
            spam_list = np.append(spam_list, spam_count)
            ham_list = np.append(ham_list, ham_count)
            spam_count = 0
            ham_count = 0
        
        mydict = {
            "Words": word_list,
            "Spam Count": spam_list,
            "Ham Count": ham_list
        }
        
        self.df2 = pd.DataFrame(mydict)
        self.df2['Spam Count'] = self.df2['Spam Count'] + 1  # Laplace smoothing
        self.df2['Ham Count'] = self.df2['Ham Count'] + 1    # Laplace smoothing
        self.df2['Spam Count Prob'] = self.df2['Spam Count'] / self.df2['Spam Count'].sum()
        self.df2['Ham Count Prob'] = self.df2['Ham Count'] / self.df2['Ham Count'].sum()
        
        self.prob_ham = len(df[df['label'] == 'ham']) / len(df)
        self.prob_spam = len(df[df['label'] == 'spam']) / len(df)
        self.words = words
    
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
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'df2': self.df2,
                'prob_ham': self.prob_ham,
                'prob_spam': self.prob_spam,
                'words': self.words
            }, f)
    
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

# Example usage:

# Train and save the model (only need to do this once)
if not os.path.exists('naive_bayes_model.pkl'):
    classifier = NaiveBayesClassifier()
    classifier.train('email_classification.csv')
    classifier.save_model('naive_bayes_model.pkl')

# Load the model and make predictions
classifier = NaiveBayesClassifier.load_model('naive_bayes_model.pkl')

email1 = """Your account has been blocked. Congratulations you have won 100 dollars for free"""
email2 = """Congradulations you have won 10000 dollars."""

spam_prob, ham_prob = classifier.predict(email1)
print(f"Email 1 - Spam Probability: {spam_prob:.2f}%, Ham Probability: {ham_prob:.2f}%")

spam_prob, ham_prob = classifier.predict(email2)
print(f"Email 2 - Spam Probability: {spam_prob:.2f}%, Ham Probability: {ham_prob:.2f}%")