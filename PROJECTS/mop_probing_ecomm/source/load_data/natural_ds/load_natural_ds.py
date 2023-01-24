import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers.readers import InputExample



def get_natural_dataset(path='/content/train.csv.zip'):
  df_natural = pd.read_csv(path)
  df_natural = df_natural.sample(frac=0.027, random_state=42)
  y_natural = df_natural['is_duplicate']
  X_natural = df_natural.drop('is_duplicate', axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X_natural, y_natural, random_state=42, test_size=0.2)
  X_train['is_duplicate'] = y_train.values
  X_test['is_duplicate'] = y_test.values
  return X_train, X_test

def get_samples_natural_dataset(df):
  samples = []
  for index, row in df.iterrows():
      label = (row['is_duplicate']) * 1.0

      guid = f"{row['id']}_{row['qid1']}_{row['qid2']}"
      
      sentence1 = row["question1"]
      sentence2 = row["question2"] 

      inp_example = InputExample(guid=guid, texts=[sentence1, sentence2], label=label)
      samples.append(inp_example)
  return samples