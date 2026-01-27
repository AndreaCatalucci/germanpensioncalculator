import numpy as np
from params import Params
from scenario_rurup_broker import ScenarioRurupBroker

def test_compounding_verification():
    p = Params()
    # Fixed parameters for predictable results
    p.years_accum = 10
    p.annual_contribution = 10000
    p.fund_fee = 0.01
    p.pension_fee = 0.005
    p.tax_working = 0.4
    p.initial_rurup = 0
    p.initial_broker = 0
    p.initial_l3 = 0
    
    # 5% constant return
    eq_returns = np.full(p.years_accum + 1, 0.05)
    
    scenario = ScenarioRurupBroker(p)
    pot = scenario.accumulate(eq_returns=eq_returns)
    
    print(f"Results for {p.years_accum} years of accumulation AFTER FIX:")
    print(f"Rurup Pot: {pot.rurup:,.2f}")
    
    # Correct calculation (10 growth steps, multiplicative fees)
    correct_rp = 0.0
    ann_c = 10000.0
    growth_factor = (1 + 0.05) * (1 - 0.01) * (1 - 0.005)
    for i in range(10):
        # The code grows the existing pot THEN adds the new contribution
        correct_rp *= growth_factor
        correct_rp += ann_c
        ann_c *= 1.02
    
    print(f"Expected Calculation (10 growth steps, multiplicative fees): {correct_rp:,.2f}")
    
    if abs(pot.rurup - correct_rp) < 1.0:
        print("SUCCESS: Rurup calculation is now accurate and off-by-one error is fixed.")
    else:
        print(f"FAILURE: Logic mismatch. Code: {pot.rurup}, Expected: {correct_rp}")
        print(f"Difference: {pot.rurup - correct_rp:,.2f}")

if __name__ == "__main__":
    test_compounding_verification()
