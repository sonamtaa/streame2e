import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import wandb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, roc_curve, confusion_matrix, classification_report

landslide_data = pd.read_csv('bhutan_landslide_data.csv')

# handle unecessary data
landslide_data.drop(['FID'], axis = 1, inplace = True)
landslide_data.drop(['STI'], axis = 1, inplace = True)
landslide_data.drop(['Type'], axis = 1, inplace = True)

#split data into x and y data
x_data = landslide_data.drop('Code',axis=1)
y_data = landslide_data['Code']

X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2, stratify = y_data, random_state=0)

# Train and predict with the model

model = RandomForestClassifier(n_estimators=100, max_features=3, min_samples_split=2,min_samples_leaf=1, max_depth=30, bootstrap=True)
# wandb.login()
# wandb.relogin()
# run = wandb.init(project='landslideone', entity='sonamtaa', name='Random Forest')

#Train the model using the training sets
model.fit(X_train, y_train)

#Predict the response for test dataset
predictions = model.predict(X_test)
# accuracy score
accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
# wandb.log({"Accuracy": accuracy , "Precision":  precision, "Recall": recall})
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)

#confusion metrix
confusion_metrix = confusion_matrix(y_test,predictions)
print('Confusion_metrix')
print(confusion_metrix)

classification = classification_report(y_test, predictions)

sns.heatmap(
  confusion_metrix,
  annot=True,
  cmap='Blues',
  fmt='g',
  xticklabels=['Predicted 0', 'Predicted 1'],
  yticklabels=['Actual 0', 'Actual 1']
  )

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.savefig('prediction_results.png', dpi=120)

model_roc_auc_rf= roc_auc_score(y_test,predictions)
print('Area Under curve: ', model_roc_auc_rf, '\n')

probabilities_rf = model.predict_proba(X_test)
fpr_rf,tpr_rf,thresholds_rf= roc_curve(y_test,probabilities_rf[:,1])

plt.plot(fpr_rf, tpr_rf)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('roc_results.png', dpi=120)

with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nAccuracy = {accuracy}, Precision of model = {precision}, Recall ofr model = {recall}.')
