# %%
import numpy as np
import pandas as pd

df = pd.read_csv("worldcities.csv")


# %%
data_table = (
    df[df["country"].isin(df["country"].value_counts().head(15).index.values)]
    .sample(frac=1)
    .groupby("country")
    .head(2000)
)

data_table = data_table[["city", "country"]]
data_table["country"].value_counts()
# %% Remove punctuation

data_table["city"] = data_table["city"].str.replace(r"[^\w\s]", "")

# %% Train tes split
from sklearn.model_selection import train_test_split

train, test = train_test_split(data_table, test_size=0.2, random_state=42)

# %% Save to csv
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

# %%
names = data_table["city"].values

# idenitify the unique characters in the dataset
unique_chars = list(set("".join(names)))
unique_chars = np.sort(unique_chars)
unique_chars = np.insert(unique_chars, 0, " ")
unique_chars
# %% Save
import json

with open("unique_chars.json", "w") as f:
    json.dump(unique_chars.tolist(), f)

# %%
