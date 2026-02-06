
import pandas as pd
print("Hello")
dfbasic = pd.read_csv("CrashData_test_6478750435646127290.csv")
dfdetailed = pd.read_csv("CrashData_test_309100363578630694 (1).csv")

merged = pd.merge(dfbasic, dfdetailed, on="Document_Nbr", how="left")

loudoun = merged[merged["Physical Juris Name"].str.contains("Loudoun", case=False, na=False)]

# Save to new CSV
loudoun.to_csv("loudoun_county.csv", index=True)