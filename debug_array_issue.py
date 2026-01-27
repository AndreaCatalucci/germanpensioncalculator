from params import Params
from scenario_broker import ScenarioBroker
from simulation import accumulate_initial_pots, sample_from_windows, collect_return_windows
from historical_returns import get_bond_returns, get_equity_returns
import numpy as np

def reproduce():
    p = Params()
    # Mock some high but not extreme returns
    eq_returns = np.array([0.15] * (p.years_accum + 1))
    bd_returns = np.array([0.05] * (p.years_accum + 1))
    
    print(f"Testing with years_accum={p.years_accum}")
    print(f"Initial broker: {p.initial_broker}")
    print(f"Annual contribution: {p.annual_contribution}")
    
    init_pot = accumulate_initial_pots(p, eq_returns, bd_returns)
    print(f"Grown initial pot: {init_pot.br_eq:,.0f}")
    
    scenario = ScenarioBroker(p)
    scen_pot = scenario.accumulate(eq_returns, bd_returns)
    print(f"Grown contributions pot: {scen_pot.br_eq:,.0f}")
    
    combined = init_pot.br_eq + scen_pot.br_eq
    print(f"Combined pot at retirement: {combined:,.0f}")

    # Now let's try with the actual synthetic data logic if no data
    # or the actual data we found
    eq_mean = 0.13
    eq_std = 0.16
    # Sample 1000 times and see how many hit the limit
    hits = 0
    max_val = 0
    for i in range(100):
        # We need p.years_accum + 1 returns
        sample_eq = np.random.normal(eq_mean, eq_std, p.years_accum + 1)
        # Cap like in synthetic logic
        sample_eq = np.clip(sample_eq, -0.5, 0.9)
        
        i_pot = accumulate_initial_pots(p, sample_eq, bd_returns)
        s_pot = scenario.accumulate(sample_eq, bd_returns)
        total = i_pot.br_eq + s_pot.br_eq
        if total > 1e8:
            hits += 1
        max_val = max(max_val, total)
        
    print(f"\nSynthetic test (100 runs):")
    print(f"Hits > 100M: {hits}")
    print(f"Max value seen: {max_val:,.0f}")

if __name__ == "__main__":
    reproduce()