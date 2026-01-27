import pickle
import os
import pandas as pd
import numpy as np

CACHE_DIR = "data_cache"
EQUITY_CACHE = os.path.join(CACHE_DIR, "equity_returns.pkl")
BOND_CACHE = os.path.join(CACHE_DIR, "bond_returns.pkl")

def check_data():
    if os.path.exists(EQUITY_CACHE):
        with open(EQUITY_CACHE, "rb") as f:
            eq = pickle.load(f)
        print(f"Equity returns (first 5):\n{eq.head()}")
        print(f"Equity count: {len(eq)}")
        print(f"Equity mean: {eq.mean():.4f}")
        print(f"Equity std: {eq.std():.4f}")
    else:
        print("Equity cache not found")

    if os.path.exists(BOND_CACHE):
        with open(BOND_CACHE, "rb") as f:
            bd = pickle.load(f)
        print(f"\nBond returns (first 5):\n{bd.head()}")
        print(f"Bond count: {len(bd)}")
        print(f"Bond mean: {bd.mean():.4f}")
        print(f"Bond std: {bd.std():.4f}")
    else:
        print("Bond cache not found")

if __name__ == "__main__":
    check_data()
