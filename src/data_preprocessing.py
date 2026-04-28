import pandas as pd

# load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# add labels
fake["label"] = 0
true["label"] = 1

# combine
df = pd.concat([fake, true], axis=0)

# shuffle
df = df.sample(frac=1, random_state=42)

# keep only required columns
df = df[['text', 'label']]

# save combined dataset
df.to_csv("data/news.csv", index=False)

print("Dataset combined successfully!")
print(df.head())