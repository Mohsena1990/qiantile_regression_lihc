import pandas as pd



path = r".\preprocessed_data.csv"
df = pd.read_csv(path)


print(df.info())