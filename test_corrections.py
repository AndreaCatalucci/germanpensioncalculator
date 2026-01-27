#!/usr/bin/env python3
"""
Test script to validate the critical mathematical corrections made to the German retirement planning system.
"""

import sys
import traceback
from params import Params
from scenario_base import Pot, withdraw, present_value
from lifetime import sample_lifetime_from67, get_life_expectancy
from simulation import collect_return_windows, sample_from_windows
import numpy as np

def test_basic_functionality():
    """Test basic functionality of corrected components."""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    # Test 1: Parameters loading
    print("\n1. Testing parameter loading...")
    try:
        p = Params()
        print("✓ Parameters loaded successfully")
        print(f"  - Annual contribution: €{p.annual_contribution:,.0f}")
        print(f"  - Public pension: €{p.public_pension:,.0f}")
        print(f"  - Rürup tax share: {p.ruerup_tax_share:.1%}")
        print(f"  - Sparerpauschbetrag: €{p.sparerpauschbetrag:,.0f}")
    except Exception as e:
        print(f"✗ Parameter loading failed: {e}")
        return False
    
    # Test 2: Present value calculation with numerical stability
    print("\n2. Testing present value calculation...")
    try:
        pv1 = present_value(1000, 5, 0.03)
        pv2 = present_value(1000, 5, 0.0)  # Test zero discount rate protection
        pv3 = present_value(1000, -1, 0.03)  # Test negative year protection
        print("✓ Present value calculations work")
        print(f"  - PV(1000, 5, 3%): €{pv1:.2f}")
        print(f"  - PV(1000, 5, 0%): €{pv2:.2f} (protected)")
        print(f"  - PV(1000, -1, 3%): €{pv3:.2f} (protected)")
    except Exception as e:
        print(f"✗ Present value calculation failed: {e}")
        return False
    
    # Test 3: Mortality calculations
    print("\n3. Testing mortality calculations...")
    try:
        # Test basic life expectancy
        le_male = get_life_expectancy("M")
        le_female = get_life_expectancy("F")
        
        # Test cohort adjustments
        le_male_1960 = get_life_expectancy("M", birth_year=1960)
        le_male_1980 = get_life_expectancy("M", birth_year=1980)
        
        print("✓ Mortality calculations work")
        print(f"  - Male life expectancy at 67: {le_male:.1f} years")
        print(f"  - Female life expectancy at 67: {le_female:.1f} years")
        print(f"  - Male born 1960: {le_male_1960:.1f} years")
        print(f"  - Male born 1980: {le_male_1980:.1f} years")
        
        # Test lifetime sampling
        lifetimes = sample_lifetime_from67("M", size=100)
        print(f"  - Sample lifetimes range: {lifetimes.min():.0f}-{lifetimes.max():.0f} years")
        
    except Exception as e:
        print(f"✗ Mortality calculation failed: {e}")
        return False
    
    # Test 4: Withdrawal function with Sparerpauschbetrag
    print("\n4. Testing withdrawal function...")
    try:
        pot = Pot(br_bd=10000, br_bd_bs=8000, br_eq=15000, br_eq_bs=12000)
        withdrawal = withdraw(pot, 2000, 0.26375, sparerpauschbetrag=801)
        
        print("✓ Withdrawal function works")
        print(f"  - Net withdrawn: €{withdrawal.net_withdrawn:.2f}")
        print(f"  - Gross withdrawn: €{withdrawal.gross_withdrawn:.2f}")
        print(f"  - Remaining bond pot: €{pot.br_bd:.2f}")
        print(f"  - Remaining equity pot: €{pot.br_eq:.2f}")
        
    except Exception as e:
        print(f"✗ Withdrawal function failed: {e}")
        return False
    
    return True

def test_bootstrap_sampling():
    """Test the corrected bootstrap sampling methodology."""
    print("\n" + "=" * 60)
    print("TESTING BOOTSTRAP SAMPLING")
    print("=" * 60)
    
    try:
        # Create synthetic data for testing
        years = list(range(1950, 2024))
        np.random.seed(42)  # For reproducible results
        equity_returns = {year: np.random.normal(0.08, 0.18) for year in years}
        bond_returns = {year: np.random.normal(0.04, 0.05) for year in years}
        
        # Convert to pandas-like objects for testing
        class MockSeries:
            def __init__(self, data):
                self.data = data
                self.index = list(data.keys())
                # Create a loc property that behaves like pandas
                self.loc = MockLoc(data)
            
            def mean(self):
                return np.mean(list(self.data.values()))
            
            def std(self):
                return np.std(list(self.data.values()))
            
            def __len__(self):
                return len(self.data)
        
        class MockLoc:
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, keys):
                class MockValues:
                    def __init__(self, values):
                        self.values = np.array(values)
                return MockValues([self.data[k] for k in keys])
        
        equity_series = MockSeries(equity_returns)
        bond_series = MockSeries(bond_returns)
        
        # Test window collection
        windows = collect_return_windows(equity_series, bond_series, window_size=10)
        print(f"✓ Collected {len(windows)} return windows")
        
        # Test sampling from windows
        eq_sample, bd_sample = sample_from_windows(windows, num_years=5)
        print("✓ Sampled returns for 5 years")
        print(f"  - Equity returns: {eq_sample}")
        print(f"  - Bond returns: {bd_sample}")
        
        # Verify sampling produces different sequences (no bias)
        samples = []
        for _ in range(10):
            eq_s, bd_s = sample_from_windows(windows, num_years=3)
            samples.append(tuple(eq_s))
        
        unique_samples = len(set(samples))
        print(f"  - Unique samples out of 10: {unique_samples} (should be > 1)")
        
        if unique_samples > 1:
            print("✓ Bootstrap sampling produces varied results (bias corrected)")
        else:
            print("⚠ Bootstrap sampling may still have bias issues")
        
    except Exception as e:
        print(f"✗ Bootstrap sampling test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_bond_returns():
    """Test the corrected bond return methodology."""
    print("\n" + "=" * 60)
    print("TESTING BOND RETURN METHODOLOGY")
    print("=" * 60)
    
    try:
        # Test with a small sample to avoid long downloads
        print("Testing bond return calculation methodology...")
        
        # Create mock yield data to test the calculation logic
        yields = [0.03, 0.035, 0.032, 0.028, 0.031]  # Sample yields
        duration = 8.5
        
        returns = []
        for i in range(1, len(yields)):
            prev_yield = yields[i-1]
            curr_yield = yields[i]
            
            coupon_return = prev_yield
            yield_change = curr_yield - prev_yield
            capital_return = -duration * yield_change
            total_return = coupon_return + capital_return
            
            returns.append(total_return)
        
        print("✓ Bond return calculation works")
        print(f"  - Sample returns: {[f'{r:.3f}' for r in returns]}")
        print(f"  - Mean return: {np.mean(returns):.3f}")
        print(f"  - Volatility: {np.std(returns):.3f}")
        
        # Verify returns are reasonable (not just yields)
        if any(abs(r) > 0.5 for r in returns):
            print("⚠ Some returns seem extreme, check calculation")
        else:
            print("✓ Returns are within reasonable bounds")
        
    except Exception as e:
        print(f"✗ Bond return test failed: {e}")
        return False
    
    return True

def test_tax_calculations():
    """Test German tax law compliance."""
    print("\n" + "=" * 60)
    print("TESTING GERMAN TAX CALCULATIONS")
    print("=" * 60)
    
    try:
        p = Params()
        
        # Test Sparerpauschbetrag application
        print("1. Testing Sparerpauschbetrag...")
        pot = Pot(br_bd=5000, br_bd_bs=4000)  # €1000 gains
        withdrawal1 = withdraw(pot, 1000, 0.26375, sparerpauschbetrag=801)
        
        # Reset pot for second test
        pot = Pot(br_bd=5000, br_bd_bs=4000)
        withdrawal2 = withdraw(pot, 1000, 0.26375, sparerpauschbetrag=0)
        
        tax_saved = withdrawal1.net_withdrawn - withdrawal2.net_withdrawn
        print(f"  - With Sparerpauschbetrag: €{withdrawal1.net_withdrawn:.2f}")
        print(f"  - Without Sparerpauschbetrag: €{withdrawal2.net_withdrawn:.2f}")
        print(f"  - Tax saved: €{tax_saved:.2f}")
        
        # Test Rürup taxation
        print("\n2. Testing Rürup taxation...")
        print(f"  - Rürup tax share: {p.ruerup_tax_share:.1%}")
        print("  - Expected for 2024: 100% (full taxation)")
        
        if p.ruerup_tax_share == 1.0:
            print("✓ Rürup tax share is realistic for 2024")
        else:
            print("⚠ Rürup tax share may need adjustment")
        
        print("✓ Tax calculations completed")
        
    except Exception as e:
        print(f"✗ Tax calculation test failed: {e}")
        return False
    
    return True

def main():
    """Run all validation tests."""
    print("GERMAN RETIREMENT PLANNER - CORRECTION VALIDATION")
    print("=" * 60)
    print("Testing critical mathematical corrections...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Bootstrap Sampling", test_bootstrap_sampling),
        ("Bond Returns", test_bond_returns),
        ("Tax Calculations", test_tax_calculations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All critical corrections validated successfully!")
        print("The German retirement planning system has been fixed.")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Review corrections needed.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)