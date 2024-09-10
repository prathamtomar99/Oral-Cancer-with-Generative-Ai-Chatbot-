from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

preds_model1 = model1.predict(X_train)
preds_model2 = model2.predict(X_train)
preds_model3 = model3.predict(X_train)

# Stack the predictions (inputs for meta-learner)
stacked_preds = np.column_stack((preds_model1, preds_model2, preds_model3))

# Split the data for meta-learner training
X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(stacked_preds, y_train, test_size=0.2, random_state=42)

# Train a simple meta-learner (e.g., Random Forest, Logistic Regression)
meta_learner = RandomForestClassifier()
meta_learner.fit(X_train_meta, y_train_meta)

# Get predictions on validation set (or test set)
test_preds_model1 = model1.predict(X_test)
test_preds_model2 = model2.predict(X_test)
test_preds_model3 = model3.predict(X_test)

stacked_test_preds = np.column_stack((test_preds_model1, test_preds_model2, test_preds_model3))

# Final predictions from meta-learner
final_predictions = meta_learner.predict(stacked_test_preds)
