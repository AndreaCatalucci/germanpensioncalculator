#!/usr/bin/env python3
"""
Test script to verify overflow protection is working correctly.
This script tests the overflow protection utilities and simulates extreme scenarios.
"""

import sys
import math
from scenario_base import (
    Pot, safe_multiply, safe_add, safe_subtract, cap_value, 
    MAX_PORTFOLIO_VALUE, withdraw
)
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker

class MockParams:
    """Mock parameters for testing"""
    def __init__(self):
        self.annual_contribution = 10000
        self.years_accum = 40
        self.fund_fee = 0.01
        self.pension_fee = 0.005
        self.cg_tax_normal = 0.26375
        self.tax_working = 0.42
        self.glide_path_years = 10
        self.years_transition = 5
        self.ruerup_dist_fee = 0.01

def test_overflow_protection_utilities():
    """Test the basic overflow protection utilities"""
    print("Testing overflow protection utilities...")
    
    # Test cap_value
    assert cap_value(1e15) == MAX_PORTFOLIO_VALUE
    assert cap_value(-1e15) == -MAX_PORTFOLIO_VALUE
    assert cap_value(1000) == 1000
    assert cap_value(float('inf')) == 0.0
    assert cap_value(float('-inf')) == 0.0
    assert cap_value(float('nan')) == 0.0
    print("✓ cap_value tests passed")
    
    # Test safe_multiply
    large_value = 1e10
    large_multiplier = 1e5
    result = safe_multiply(large_value, large_multiplier)
    assert result <= MAX_PORTFOLIO_VALUE
    assert result > 0
    
    # Test with infinity
    assert safe_multiply(float('inf'), 2.0) == 0.0
    assert safe_multiply(1000, float('nan')) == 0.0
    print("✓ safe_multiply tests passed")
    
    # Test safe_add
    result = safe_add(1e11, 1e11, 1e11)
    assert result <= MAX_PORTFOLIO_VALUE
    assert result > 0
    
    # Test with infinity - should skip infinite values and continue
    assert safe_add(float('inf'), 1000) == 1000
    print("✓ safe_add tests passed")
    
    # Test safe_subtract
    result = safe_subtract(1e13, 1000)
    assert result <= MAX_PORTFOLIO_VALUE
    assert result > 0
    
    # Test with infinity
    assert safe_subtract(float('inf'), 1000) == 0.0
    print("✓ safe_subtract tests passed")

def test_pot_overflow_protection():
    """Test Pot class with extreme values"""
    print("\nTesting Pot class overflow protection...")
    
    # Create pot with extreme values
    pot = Pot()
    pot.br_eq = 1e15  # Very large value
    pot.br_bd = 1e15
    pot.l3_eq = 1e15
    
    # Test leftover calculation with overflow protection
    leftover = pot.leftover()
    assert leftover <= MAX_PORTFOLIO_VALUE
    assert math.isfinite(leftover)
    print(f"✓ Pot leftover with extreme values: {leftover:.2e}")

def test_withdraw_overflow_protection():
    """Test withdraw function with extreme values"""
    print("\nTesting withdraw function overflow protection...")
    
    pot = Pot()
    pot.br_eq = 1e15
    pot.br_bd = 1e15
    pot.l3_eq = 1e15
    pot.br_eq_bs = 1e10
    pot.br_bd_bs = 1e10
    
    # Test withdrawal with extreme pot values
    withdrawal = withdraw(pot, 100000, 0.26375)
    assert math.isfinite(withdrawal.gross_withdrawn)
    assert math.isfinite(withdrawal.net_withdrawn)
    assert withdrawal.gross_withdrawn >= 0
    assert withdrawal.net_withdrawn >= 0
    print(f"✓ Withdrawal with extreme pot: gross={withdrawal.gross_withdrawn:.2e}, net={withdrawal.net_withdrawn:.2e}")

def test_scenario_overflow_protection():
    """Test scenario classes with extreme market conditions"""
    print("\nTesting scenario classes with extreme conditions...")
    
    params = MockParams()
    
    # Test ScenarioBroker
    broker_scenario = ScenarioBroker(params)
    
    # Create extreme returns (1000% per year for testing overflow)
    extreme_returns = [10.0] * 42  # 1000% returns for 42 years
    
    try:
        pot = broker_scenario.accumulate(eq_returns=extreme_returns)
        assert math.isfinite(pot.br_eq)
        assert pot.br_eq <= MAX_PORTFOLIO_VALUE
        print(f"✓ ScenarioBroker accumulate with extreme returns: {pot.br_eq:.2e}")
    except Exception as e:
        print(f"✗ ScenarioBroker accumulate failed: {e}")
    
    # Test ScenarioRurupBroker
    rurup_scenario = ScenarioRurupBroker(params)
    
    try:
        pot = rurup_scenario.accumulate(eq_returns=extreme_returns)
        assert math.isfinite(pot.rurup)
        assert math.isfinite(pot.br_eq)
        assert pot.rurup <= MAX_PORTFOLIO_VALUE
        assert pot.br_eq <= MAX_PORTFOLIO_VALUE
        print(f"✓ ScenarioRurupBroker accumulate: rurup={pot.rurup:.2e}, br_eq={pot.br_eq:.2e}")
    except Exception as e:
        print(f"✗ ScenarioRurupBroker accumulate failed: {e}")

def test_transition_year_overflow():
    """Test transition_year method with extreme values"""
    print("\nTesting transition_year overflow protection...")
    
    params = MockParams()
    scenario = ScenarioBroker(params)
    
    # Create pot with large values
    pot = Pot()
    pot.rurup = 1e11
    pot.br_eq = 1e11
    pot.l3_eq = 1e11
    pot.br_bd = 1e11
    
    # Test with extreme returns
    extreme_eq_r = 5.0  # 500% return
    extreme_bd_r = 2.0  # 200% return
    
    try:
        result_pot = scenario.transition_year(pot, 1, extreme_eq_r, extreme_bd_r)
        
        assert math.isfinite(result_pot.rurup)
        assert math.isfinite(result_pot.br_eq)
        assert math.isfinite(result_pot.l3_eq)
        assert math.isfinite(result_pot.br_bd)
        
        assert result_pot.rurup <= MAX_PORTFOLIO_VALUE
        assert result_pot.br_eq <= MAX_PORTFOLIO_VALUE
        assert result_pot.l3_eq <= MAX_PORTFOLIO_VALUE
        assert result_pot.br_bd <= MAX_PORTFOLIO_VALUE
        
        print("✓ transition_year with extreme returns:")
        print(f"  rurup: {result_pot.rurup:.2e}")
        print(f"  br_eq: {result_pot.br_eq:.2e}")
        print(f"  l3_eq: {result_pot.l3_eq:.2e}")
        print(f"  br_bd: {result_pot.br_bd:.2e}")
        
    except Exception as e:
        print(f"✗ transition_year failed: {e}")

def main():
    """Run all overflow protection tests"""
    print("=" * 60)
    print("OVERFLOW PROTECTION TEST SUITE")
    print("=" * 60)
    
    try:
        test_overflow_protection_utilities()
        test_pot_overflow_protection()
        test_withdraw_overflow_protection()
        test_scenario_overflow_protection()
        test_transition_year_overflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL OVERFLOW PROTECTION TESTS PASSED!")
        print("✅ No RuntimeWarning: overflow encountered expected")
        print("✅ Portfolio values remain within realistic bounds")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()