import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string

# Loading datasets
data_fake = pd.read_csv('C:/Users/91824/OneDrive/Desktop/Fake_news/Fake.csv')
data_true = pd.read_csv('C:/Users/91824/OneDrive/Desktop/Fake_news/True.csv')

# Adding a class column
data_fake['class'] = 0
data_true['class'] = 1

# Separating last 10 rows for manual testing
data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

# Dropping the last 10 rows from original datasets
data_fake = data_fake.iloc[:-10]
data_true = data_true.iloc[:-10]

# Assigning class labels to manual test data
data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

# Merging datasets
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Shuffling and resetting index
data = data.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying preprocessing on text column
data['text'] = data['text'].apply(wordopt)

# Splitting dataset
x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Model training
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GB = GradientBoostingClassifier()
GB.fit(xv_train, y_train)

RF = RandomForestClassifier()
RF.fit(xv_train, y_train)

# Helper function to return class label
def output(n):
    return "FAKE" if n == 0 else "TRUE"

# Manual test function for predicting on new inputs
def manual_test(news):
    # Create a DataFrame for the input news text
    testing_news = {"text": [news]}
    new_def_text = pd.DataFrame(testing_news)

    # Preprocess the input text
    new_def_text["text"] = new_def_text["text"].apply(wordopt)

    # Transform the text using the trained TF-IDF Vectorizer
    new_xv_test = vectorization.transform(new_def_text["text"])

    # Model predictions
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    # Formatting predictions as a dictionary
    result = {
        "Logistic Regression": output(pred_LR[0]),
        "Decision Tree": output(pred_DT[0]),
        "Gradient Boosting": output(pred_GB[0]),
        "Random Forest": output(pred_RF[0])
    }

    return result

# Example usage for manual testing
if __name__ == "__main__":
    # Example input text to test
    input_text = "A freak road accident in Kerala resulted in the death of a two-year-old girl who was travelling with her mother, police said. The girl, who was sitting in her mother's lap, died of suffocation after the airbags opened following an accident. The incident occurred when the family’s car collided with a tanker lorry in the Kottakkal-Padaparambu area."

    predictions = manual_test(input_text)
    print("Prediction Results:")
    for model, result in predictions.items():
        print(f"{model}: {result}")
