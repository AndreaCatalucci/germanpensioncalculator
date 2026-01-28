
import unittest
from scenario_base import calculate_vorabpauschale, withdraw, Pot, Withdrawal, safe_add, safe_subtract
from params import Params

class TestTaxOptimization(unittest.TestCase):
    def test_vorabpauschale(self):
        # Case 1: Standard Gain > VP
        # Start: 100, End: 110. Gain: 10.
        # Basiszins: 2.5%. Reference: 100 * 0.025 * 0.7 = 1.75
        # Taxable Base: min(10, 1.75) = 1.75
        # Net Taxable (TFS 30%): 1.75 * 0.7 = 1.225
        res = calculate_vorabpauschale(100, 110, 0.025, 0.30)
        self.assertAlmostEqual(res, 1.225)

        # Case 2: Gain < VP but > 0
        # Start: 100, End: 101. Gain: 1.
        # Reference: 1.75.
        # Taxable Base: min(1, 1.75) = 1.0
        # Net Taxable: 0.7
        res = calculate_vorabpauschale(100, 101, 0.025, 0.30)
        self.assertAlmostEqual(res, 0.7)
        
        # Case 3: Loss
        # Start: 100, End: 90.
        # Base: 0
        res = calculate_vorabpauschale(100, 90, 0.025, 0.30)
        self.assertEqual(res, 0.0)

    def test_withdraw_l3_prioritization(self):
        # Setup pot
        # Broker Bond: 10k (Basis 10k -> No Gain)
        # Broker Equity: 50k (Basis 10k -> 40k Gain)
        # L3 Equity: 50k (Basis 10k -> 40k Gain)
        
        # Taxes: 26.375% Broker. 
        # L3: 30% personal rate, 15% TFS, Half-Income.
        # Effective L3 Tax = Gain * 0.85 * 0.5 * 0.30 = Gain * 0.1275 ~ 12.75%
        
        pot = Pot(
            br_bd=10000, br_bd_bs=10000,
            br_eq=50000, br_eq_bs=10000,
            l3_eq=50000, l3_eq_bs=10000
        )
        
        cg_tax = 0.26375
        # Sparerpauschbetrag = 1000.
        
        # Need 20,000 net.
        # 1. Broker Bond (10k). All principal, so 10k net withdrawn. Remaining need: 10k.
        # 2. Broker Equity (Allowance). 1000 allowance.
        #    Gain Ratio = 40k/50k = 0.8.
        #    Gross needed for 1000 allowance = 1000 / 0.8 = 1250.
        #    This gives 1250 net (tax free). Remaining need: 8750.
        # 3. L3 Equity (Cheaper tax).
        #    Effective tax rate ~ 12.75%.
        #    Should withdraw from here.
        
        w = withdraw(pot, 20000, cg_tax=cg_tax, sparerpauschbetrag=1000)
        
        # Check that pot.br_bd is 0 (Used first)
        self.assertAlmostEqual(pot.br_bd, 0.0)
        
        # Check that pot.br_eq reduced by approx 1250 (Allowance usage)
        # It shouldn't be emptied because L3 is cheaper!
        self.assertGreater(pot.br_eq, 40000) 
        
        # Check that pot.l3_eq reduced significantly (Filling the rest)
        self.assertLess(pot.l3_eq, 45000)

    def test_withdraw_l3_calculation(self):
        # Test specific L3 tax calc
        pot = Pot(l3_eq=10000, l3_eq_bs=0) # 100% Gain
        # Taxable = 10000 * 0.85 * 0.5 = 4250.
        # Tax (30%) = 1275.
        # Net available = 10000 - 1275 = 8725.
        
        # Withdraw all
        w = withdraw(pot, 20000, 0.25, sparerpauschbetrag=0, tax_retirement=0.30, tfs_insurance=0.15)
        
        self.assertAlmostEqual(w.net_withdrawn, 8725)
        self.assertAlmostEqual(w.gross_withdrawn, 10000)
        self.assertAlmostEqual(pot.l3_eq, 0.0)

if __name__ == '__main__':
    unittest.main()
