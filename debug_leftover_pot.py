#!/usr/bin/env python3
"""
Debug script to investigate the huge leftover pot issue.
"""

from scenario_safe_spend import ScenarioSafeSpend
from params import Params
from simulation import collect_return_windows, sample_from_windows
from historical_returns import get_equity_returns, get_bond_returns

def debug_pot_growth():
    """Debug the pot growth to find where values explode."""
    print("DEBUGGING LEFTOVER POT ISSUE")
    print("=" * 50)
    
    p = Params()
    p.num_sims = 1  # Single simulation for debugging
    scenario = ScenarioSafeSpend(p)
    
    # Get historical data
    equity_returns = get_equity_returns(
        series_id=p.equity_series_id, start_year=p.data_start_year
    )
    bond_returns = get_bond_returns(
        series_id=p.bond_series_id, start_year=p.data_start_year
    )
    
    # Collect windows
    windows = collect_return_windows(
        equity_returns, bond_returns, p.bootstrap_period_length
    )
    
    # Sample returns for accumulation phase
    accumulation_eq_returns, accumulation_bd_returns = sample_from_windows(
        windows, p.years_accum + 1
    )
    
    print(f"Accumulation phase returns (first 10): {accumulation_eq_returns[:10]}")
    print(f"Bond returns (first 10): {accumulation_bd_returns[:10]}")
    
    # Test scenario accumulation
    print("\nTesting scenario accumulation...")
    scen_pot = scenario.accumulate(accumulation_eq_returns, accumulation_bd_returns)
    print("After accumulation:")
    print(f"  - Rürup: €{scen_pot.rurup:,.2f}")
    print(f"  - L3 equity: €{scen_pot.l3_eq:,.2f}")
    print(f"  - Broker equity: €{scen_pot.br_eq:,.2f}")
    print(f"  - Broker bonds: €{scen_pot.br_bd:,.2f}")
    print(f"  - Total leftover: €{scen_pot.leftover():,.2f}")
    
    # Test transition phase
    print("\nTesting transition phase...")
    transition_eq_returns, transition_bd_returns = sample_from_windows(
        windows, p.years_transition
    )
    
    for year in range(min(3, p.years_transition)):  # Test first 3 years
        eq_r = float(transition_eq_returns[year])
        bd_r = float(transition_bd_returns[year])
        print(f"\nTransition year {year}: eq_r={eq_r:.4f}, bd_r={bd_r:.4f}")
        print(f"  Before: L3={scen_pot.l3_eq:,.2f}, BR_EQ={scen_pot.br_eq:,.2f}, BR_BD={scen_pot.br_bd:,.2f}")
        
        scenario.transition_year(scen_pot, year, eq_r, bd_r)
        
        print(f"  After: L3={scen_pot.l3_eq:,.2f}, BR_EQ={scen_pot.br_eq:,.2f}, BR_BD={scen_pot.br_bd:,.2f}")
        print(f"  Leftover: €{scen_pot.leftover():,.2f}")
        
        # Check for explosive growth
        if scen_pot.leftover() > 1e12:  # 1 trillion
            print("⚠️  EXPLOSIVE GROWTH DETECTED!")
            break
    
    print(f"\nFinal leftover after transition sample: €{scen_pot.leftover():,.2f}")

if __name__ == "__main__":
    debug_pot_growth()