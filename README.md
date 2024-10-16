# sms_spam_detector

sms_text_df
    
# Import pandas
import pandas as pd
# Import the required dependencies from sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Set the column width to view the text message data.
pd.set_option('max_colwidth', 200)

# Import Gradio
import gradio as gr

"""
    Perform SMS classification using a pipeline with TF-IDF vectorization and Linear Support Vector Classification.

    Parameters:
    - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns for SMS classification.

    Returns:
    - text_clf (Pipeline): Fitted pipeline model for SMS classification.

    This function takes a DataFrame with 'text_message' and 'label' columns, splits the data into
    training and testing sets, builds a pipeline with TF-IDF vectorization and Linear Support Vector
    Classification, and fits the model to the training data. 
    The fitted pipeline is returned to make future predictions.
    """
    # Set the features variable to the text message column.
    X = sms_text_df['text_message']  
    # Set the target variable to the "label" column.
    y = sms_text_df['label']

    # Split data into training and testing and set the test_size = 33%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Set the target variable to the "label" column.
    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                     ('clf', LinearSVC()),
])

    # Fit the model to the transformed data.
    text_clf.fit(X_train, y_train)  

    # Split data into training and testing and set the test_size = 33%
    

    # Build a pipeline to transform the test set to compare to the training set.
    

    # Fit the model to the transformed training data and return model.
    return text_clf
    
# Create a function called `sms_prediction` that takes in the SMS text and predicts the whether the text is "not spam" or "spam". 
# The function should return the SMS message, and say whether the text is "not spam" or "spam".
def sms_prediction(text):
    """
    Predict the spam/ham classification of a given text message using a pre-trained model.

    Parameters:
    - text (str): The text message to be classified.

    Returns:
    - str: A message indicating whether the text message is classified as spam or not.

    This function takes a text message and a pre-trained pipeline model, then predicts the
    spam/ham classification of the text. The result is a message stating whether the text is
    classified as spam or not.
    """
    # Create a variable that will hold the prediction of a new text.
    str = text_clf.predict([text])
    # Using a conditional if the prediction is "ham" return the message:
    if (str == 'ham'):
        return f'The text message: "{text}", is spam.'
    else:
        return f'The text message: "{text}", is not spam.'
    # f'The text message: "{text}", is not spam.' Else, return f'The text message: "{text}", is spam.'
    
    # Create a sms_app that takes a textbox for the inputs and has a textbox for the output.  
# Povide labels for each textbox. 

app = gr.Interface(fn=sms_prediction, 
                   inputs=["text"], outputs="text")
# Launch the app
app.launch()
# Launch the app.
text_1='You are a lucky winner of $5000!'
text_2='You won 2 free tickets to the Super Bowl.'
text_3='You won 2 free tickets to the Super Bowl text us to claim your prize.'
text_4='Thanks for registering. Text 4343 to receive free updates on medicare.'
print(sms_prediction(text_1))
print(sms_prediction(text_2))
print(sms_prediction(text_3))
print(sms_prediction(text_4))