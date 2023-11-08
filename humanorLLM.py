the full code 


```
# WHAT :  WAYST model  : Who are you storyteller ? 
# Author Dzoan nguyen 
# status : poc , draft .

# Import necessary libraries.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
import tensorflow as tf


# Load  the dataset.

train_essays1 = pd.read_csv("/kaggle/input/train_essays.csv")
train_essays2 = pd.read_csv("/kaggle/input/ai_generated_train_essays.csv")
train_essays3 = pd.read_csv("/kaggle/input/external_dataset.csv")
test_essays = pd.read_csv("/kaggle/input/test_essays.csv")

# Define constants based on your dataset.
VOCAB_SIZE = 20000  # Set based on your dataset
MAXLEN = 512  # BERT's maximum sequence length

# Tokenize the text using BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_essays['text']), truncation=True, padding='max_length', max_length=MAXLEN)
test_encodings = tokenizer(list(test_essays['text']), truncation=True, padding='max_length', max_length=MAXLEN)

# Convert to TensorFlow datasets.
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))

# Load BERT model for sequence classification.
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile the BERT model.
optimizer = Adam(learning_rate=5e-5)
bert_model.compile(optimizer=optimizer, loss=bert_model.compute_loss, metrics=['accuracy'])

# Fit the BERT model.
bert_model.fit(train_dataset.shuffle(100).batch(32), epochs=3, batch_size=8)

# Define base learners for stacking ensemble.
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
]

# Initialize the base learners.
base_models = {name: model for name, model in base_learners}

# Train each base model and store the out-of-fold predictions
# These predictions will be used as features for the meta-learner.
X_train_meta = np.zeros((len(X_train), len(base_learners)))  # Placeholder for meta features

for i, (name, model) in enumerate(base_models.items()):
    print(f"Training base model {name}...")
    model.fit(X_train, y_train)
    X_train_meta[:, i] = model.predict_proba(X_train)[:, 1]
    print(f"Model {name} trained.")

# Define meta-learner.
meta_learner = LogisticRegression()

# Train the meta-learner on the out-of-fold predictions of the base learners.
print("Training meta-learner...")
stacked_ensemble = meta_learner.fit(X_train_meta, y_train)
print("Meta-learner trained.")

# Prepare input for the stacking ensemble.
# Extract predictions from base models for the test set.
X_test_meta = np.zeros((len(X_test), len(base_learners)))  # Placeholder for meta features

for i, (name, model) in enumerate(base_models.items()):
    X_test_meta[:, i] = model.predict_proba(X_test)[:, 1]

# Predict with the meta-learner.
ensemble_preds = stacked_ensemble.predict_proba(X_test_meta)[:, 1]

# Create a submission file for the competition.
submission = pd.DataFrame({'id': test_essays.index, 'generated': ensemble_preds})
submission.to_csv("submission.csv", index=False)  # Save the submission file for competition entry.

