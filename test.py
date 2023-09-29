import pandas as pd
df = pd.DataFrame()
df[("A", "a")] = [1,2,3,4]
df[("A", "b")] = [2,3,4,5]
df[("B", "a")] = [1,2,3,4]
df[("B", "b")] = [2,3,4,5]

print(df["A"])

