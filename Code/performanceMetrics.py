import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# Read the Excel file
df = pd.read_excel('/Users/maazinzaidi/Dropbox/EME/EC-312 Digital Image Processing/Lab/Project/Results.xlsx')

# Extract predicted and ground truth values
predicted = df['Predicted'].values
ground_truth = df['Ground Truth'].values

# Create confusion matrix
confusion_mat = confusion_matrix(ground_truth, predicted)

accuracy = accuracy_score(ground_truth, predicted)

print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy: ", accuracy * 100, "%")
