import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# <<1>> Loading the model and cleaned data
model = joblib.load('credit_model.pkl')
df = pd.read_csv('data/cleaned_credit_data.csv')

# <<2>> Re-Split the model again exactly like in train.py 
x = df.drop('Risk_good', axis = 1)
# Keep the only the y col as output
y = df['Risk_good']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# <<3>> Generate the model predictions 
y_pred = model.predict(X_test)

# <<4>> Creating a confusion model to see where it getting lost
confm = confusion_matrix(y_test,y_pred)

# <<5>>. PLotting the data with seaborn
plt.figure(figsize=(8,6))
sns.heatmap(confm, annot=True, fmt='g', cmap='Greens', 
            xticklabels=['Bad (Predicted)', 'Good (Predicted)'],
            yticklabels=['Bad (Actual)', 'Good (Actual)'])

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Credit Risk Model: Confusion Matrix')

# <<6>>. Save the visual for your portfolio
plt.savefig('credit_confusion_matrix.png')
print("Visual saved as credit_confusion_matrix.png")

# <<7>>. Show the plot
plt.show()