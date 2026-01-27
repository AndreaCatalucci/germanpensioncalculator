import unittest
import numpy as np
from params import Params
from scenario_base import Pot, withdraw, shift_equity_to_bonds, Scenario
from scenario_l3_broker import ScenarioL3Broker

class TestTaxLogic(unittest.TestCase):
    def setUp(self):
        self.params = Params()
        self.params.cg_tax_normal = 0.26375
        self.params.tfs_broker = 0.30
        self.params.tfs_insurance = 0.15
        self.params.tax_retirement = 0.30
        self.params.sparerpauschbetrag = 1000

    def test_broker_withdrawal_with_tfs(self):
        """
        Test that broker withdrawal correctly applies 30% TFS.
        If gain is 1000, 300 is tax-free, 700 is taxed at 26.375%.
        Tax should be 700 * 0.26375 = 184.625.
        Net should be 1000 - 184.625 = 815.375.
        """
        pot = Pot(br_eq=2000, br_eq_bs=1000) # 1000 gain
        # We need to bypass the sparerpauschbetrag for this test or set it to 0
        w = withdraw(pot, net_needed=2000, cg_tax=0.26375, sparerpauschbetrag=0)
        
        # Expected tax: (2000 - 1000) * (1 - 0.30) * 0.26375 = 184.625
        # Expected net: 2000 - 184.625 = 1815.375
        self.assertAlmostEqual(w.net_withdrawn, 1815.375, places=2)

    def test_l3_taxation_halbeinkuenfte(self):
        """
        Test that L3 transition correctly applies Halbeinkünfteverfahren.
        Move 1000 from L3 with 500 gain. 15% TFS applies first.
        Taxable gain = (500 * (1 - 0.15)) * 0.5 = 212.5.
        Tax = 212.5 * tax_retirement (0.30) = 63.75.
        Net moved = 1000 - 63.75 = 936.25.
        """
        class MockScenario(Scenario):
            def accumulate(self, eq_returns=None, bd_returns=None): return Pot()
            def decumulate_year(self, pot, current_year, net_ann, needed_net, rand_returns): return 0, pot

        self.params.pension_fee = 0.0
        self.params.fund_fee = 0.0
        self.params.years_transition = 1
        self.params.age_retire = 67
        self.params.current_age = 67
        self.params.ruerup_dist_fee = 0.0 # Simplify
        
        scenario = MockScenario(self.params)
        pot = Pot(l3_eq=1000, l3_eq_bs=500)
        
        # Move everything in 1 year
        new_pot = scenario.transition_year(pot, current_year=0, eq_r=0, bd_r=0)
        
        # Expected tax: (1000 - 500) * (1 - 0.15) * 0.5 * 0.30 = 63.75
        # Expected br_bd: 1000 - 63.75 = 936.25
        self.assertAlmostEqual(new_pot.br_bd, 936.25, places=2)

if __name__ == '__main__':
    unittest.main()
