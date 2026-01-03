import pandas as pd
# joblib - The "Save Button" for your AI lets you save your trained model into a file without retraining.
import joblib
# XGBClassifier - The actual high-powered math model that learns to predict credit risk.
from xgboost import XGBClassifier
# train_test_split - hides 20% of your data from the AI so you can test it later to see if it actually learned.
from sklearn.model_selection import train_test_split
# " It calculates the percentage of correct predictions your model made.
from sklearn.metrics import accuracy_score, classification_report


# <<1>> Load the data
df = pd.read_csv("data/cleaned_credit_data.csv", index_col = 0)

# <<2>> Separate the Features (x) from the Target (y)
# 'Risk' is what we want to predict (1 = Good, 2 = Bad in this dataset)

# Bascially saying keep every other column as input but the risk col
x = df.drop('Risk_good', axis = 1)
# Keep the only the y col as output
y = df['Risk_good']

# <<3>> Split into Training and Testing
# X_train - The 80% of "questions" (Age, Job, etc.) the model will study.
# X_test - The 20% of "questions" kept in a vault for the final exam.
# y_train - The 80% of "answers" (Risk) the model will use to see if it guessed right during training.
# y_test - The 20% of "answers" used to grade that final exam. This all prevents OverFitting or so the model doesnt just memorize the data points and answer
# test_size - This tells the computer to take 20% of your data and set it aside for testing.
# random_state - Lets you choose a specific seed of questions if left blank the computer keeps choosing a random set of 20% questions
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# <<4>> Initialzing and training XGBoost
# Use 'binary:logistic' for Yes/No predictions or Risky or not in this case

# The argumenst in this case are called 'Hyperparameters'
# n_estimators - Tells the model to build 100 trees in a row but if you have too many (like 10,000), the model might just "memorize" the data (overfitting)
# learning_rate - how much each tree is allowed to change the overall final score to high though and the model jumps to conclusions
# max_depth -  depth of 5 means each tree can ask a 
# maximum of 5 "Yes/No" questions before making a guess (e.g., "Is Age > 30?" $\rightarrow$ "Is Savings > 500?" $\rightarrow$ etc.).
# Engineering Tip - model = XGBClassifier() The defaults are designed for general data, not specific financial risk.
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# .fit() - Finally this tells the model to study for its final 
# but dont give it the X_Test or Y_test or it will memorize the answers and cheat
model.fit(X_train,y_train)

# <<5>> Checking the accuracy or the Final Exam
predictions = model.predict(X_test)
print(f"\n ---Model Accuracy--- \n {accuracy_score(y_test,predictions):.2f}")
print("\n ---Detailed Report:--- \n")
print(classification_report(y_test,predictions))

# classification_report data
# Support - Tells you how many of each answer were in the test aka 59 bad 141 good
# Precision = Basically says how good it it as preditcing the specific outcome 
# Ex. When the model says " Good," it is right 79% of the time.
# Recall - Out of the total of the two groups how many did it catch correctly
# Ex. Out of all the good people in the data, the model successfully "caught" or found 90% of them.
# F1 Score - This is just a mathematical average of Precision and Recall. In banking, a 0.84 for "True" is solid, 
# but the 0.51 for "False" is bad
# Macro Avg - This treats "Good" and "Bad" as equally important.
# Weighted AVG - This gives more "weight" to the "True" row because there were way more good borrowers (141) than bad ones (59).


# 6. SAVE THE BRAIN (Persistence)
joblib.dump(model, 'credit_model.pkl')
print("\nModel saved as credit_model.pkl")
