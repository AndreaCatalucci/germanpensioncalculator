import numpy as np
from params import Params
from scenario_rurup_broker import ScenarioRurupBroker
from simulation import accumulate_initial_pots

def test_compounding_reproduction():
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
    
    print(f"Results for {p.years_accum} years of accumulation:")
    print(f"Rurup Pot: {pot.rurup:,.2f}")
    
    # Manual calculation (Linear fee as per current code: 1 + r - f1 - f2)
    manual_rp = 0.0
    ann_c = 10000.0
    for i in range(10):
        manual_rp *= (1 + 0.05 - 0.01 - 0.005)
        manual_rp += ann_c
        ann_c *= 1.02
    
    # Current code also does a redundant final year growth:
    manual_rp *= (1 + 0.05 - 0.01 - 0.005)
    
    print(f"Manual Calculation (11 growth steps): {manual_rp:,.2f}")
    
    if abs(pot.rurup - manual_rp) < 1.0:
        print("SUCCESS: Reproduction script matches current logic (including redundant year).")
    else:
        print(f"FAILURE: Logic mismatch. Code: {pot.rurup}, Manual: {manual_rp}")

    # Correct calculation (10 growth steps, multiplicative fees)
    correct_rp = 0.0
    ann_c = 10000.0
    growth_factor = (1 + 0.05) * (1 - 0.01) * (1 - 0.005)
    for i in range(10):
        correct_rp *= growth_factor
        correct_rp += ann_c
        ann_c *= 1.02
    
    # The current code's approach of adding contribution THEN growing (in the next loop)
    # is actually growing contributions from previous years, but the last year's contribution
    # is added and then grown by the final year logic.
    # Total years of growth for first contribution = 10 (loop) + 1 (final) = 11? No.
    
    print(f"Correct Calculation (10 growth steps, multiplicative fees): {correct_rp:,.2f}")
    print(f"Difference: {pot.rurup - correct_rp:,.2f} ({(pot.rurup/correct_rp - 1)*100:.2f}%)")

if __name__ == "__main__":
    test_compounding_reproduction()
