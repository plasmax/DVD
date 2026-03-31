import pandas as pd
import numpy as np

path = "saves/v005/loss_log.csv"
df = pd.read_csv(path)

loss_cols = [c for c in df.columns if c not in ["global_step", "learning_rate"]]

print("ROWS", len(df))
print("COLUMNS", list(df.columns))
print("\nFIRST 5")
print(df.head(5).to_string(index=False))
print("\nLAST 10")
print(df.tail(10).to_string(index=False))

print("\nSUMMARY")
print(df[loss_cols].describe().to_string())

print("\nBEST STEPS")
for col in loss_cols:
    i = df[col].idxmin()
    print(f"{col}: best={df.loc[i, col]:.8f} at step={int(df.loc[i, 'global_step'])}")

print("\nEND TREND")
for col in loss_cols:
    s10 = df[col].tail(10).mean()
    s20 = df[col].tail(20).mean() if len(df) >= 20 else df[col].mean()
    print(f"{col}: last10_mean={s10:.8f} last20_mean={s20:.8f} delta={s10 - s20:+.8f}")

print("\nDOWNSAMPLED")
n = 20
idx = np.linspace(0, len(df)-1, min(n, len(df))).round().astype(int)
print(df.iloc[np.unique(idx)].to_string(index=False))
