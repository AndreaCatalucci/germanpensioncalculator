import unittest
import numpy as np
from params import Params
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_l3_broker import ScenarioL3Broker

class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.params = Params()
        self.params.annual_contribution = 20000
        self.params.tax_working = 0.4431
        self.params.years_accum = 1 # Test with 1 year for simplicity
        self.params.fund_fee = 0.0
        self.params.pension_fee = 0.0

    def test_broker_normalization(self):
        """
        Broker should invest exactly annual_contribution.
        """
        scenario = ScenarioBroker(self.params)
        eq_returns = np.array([0.0, 0.0]) # 0% return
        pot = scenario.accumulate(eq_returns=eq_returns)
        
        # Expected: 20000 (contribution) + 0 (return in final year)
        # However, scenario_broker.py has eq_val *= 1 + final_eq_r
        # Let's check the code: it does eq_val += ann_contr, then loop ends, 
        # then final growth.
        self.assertEqual(pot.br_eq, 20000)

    def test_rurup_broker_normalization(self):
        """
        Rurup Broker should invest 20000 in Rurup AND (20000 * 0.4431) in Broker.
        """
        scenario = ScenarioRurupBroker(self.params)
        eq_returns = np.array([0.0, 0.0])
        pot = scenario.accumulate(eq_returns=eq_returns)
        
        # Rurup part: 20000
        # Broker part: 20000 * 0.4431 = 8862
        self.assertEqual(pot.rurup, 20000)
        self.assertEqual(pot.br_eq, 8862)

    def test_l3_broker_normalization(self):
        """
        L3 Broker should invest total 20000 (split 40/60).
        """
        scenario = ScenarioL3Broker(self.params)
        eq_returns = np.array([0.0, 0.0])
        bd_returns = np.array([0.0, 0.0])
        pot = scenario.accumulate(eq_returns=eq_returns, bd_returns=bd_returns)
        
        # Total = 20000
        total = pot.l3_eq + pot.br_eq + pot.br_bd
        self.assertEqual(total, 20000)

if __name__ == '__main__':
    unittest.main()
