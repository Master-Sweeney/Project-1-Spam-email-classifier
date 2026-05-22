#Import neccessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                               precision_score, recall_score,
                               classification_report)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Import email data from dataset
df = pd.read_csv(
    "/Users/alanngungi/Desktop/Important/Work/Data Eng Portfolio/Spam Email detector/spam.csv",
                 encoding='cp1252')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)

# Replace the null values with a null string
emails=df.where((pd.notnull(df)),'')

# Label Ham mail as 0 ; Spam mail as 1;
emails.loc[emails['v1'] == 'ham','v1']= 0
emails.loc[emails['v1'] == 'spam', 'v1']= 1

#Vertical Split, feature matrix and target vector
X = emails['v2']
labels= emails['v1']
labels=labels.astype(int)

#Horizontal Split, train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Vectorization: Convert text into a matrix of token counts. This creates a "vocabulary" of all words in your emails.
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_train)


#Train MultinomialNB model, a standard choice for text classification.
model_mnb = MultinomialNB()
model_mnb.fit(X_vec, y_train)
print("✅ MultinomialNB Model training complete!")

#Make predictions with test split data
X_test_vectorized = vectorizer.transform(X_test)
y_pred_mnb = model_mnb.predict(X_test_vectorized)

#MultinomialNB model performance evaluation
accuracy_mnb  = accuracy_score(y_test, y_pred_mnb)
precision_mnb = precision_score(y_test, y_pred_mnb,zero_division=0)
recall_mnb    = recall_score(y_test, y_pred_mnb,zero_division=0)
conf_mat_mnb  = confusion_matrix(y_test, y_pred_mnb)

print(f"\n Model Performance:")
print(f"   Accuracy:  {accuracy_mnb * 100:.1f}%")
print(f"   Precision: {precision_mnb * 100:.1f}%")
print(f"   Recall:    {recall_mnb * 100:.1f}%")
print(f"\nConfusion Matrix:")
print(conf_mat_mnb)
print("\nFull Report:")
print(classification_report(y_test, y_pred_mnb,
      target_names=['Ham', 'Spam']))

#Taking user input mail 

input_mail=[input("Enter a Mail : ")]

# Convert the text into feature vectors

input_data_feature=vectorizer.transform(input_mail)

# Making Predictions

prediction = model_mnb.predict(input_data_feature)
print(prediction)

if (prediction[0]== 1):
    print("Spam Mail")
else:
    print("Ham Mail")