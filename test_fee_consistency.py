import unittest
import numpy as np
from params import Params
from scenario_base import Pot
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_l3 import ScenarioL3
from scenario_l3_broker import ScenarioL3Broker
from scenario_enhanced_broker import ScenarioEnhancedBroker
from scenario_enhanced_l3_broker import ScenarioEnhancedL3Broker

class TestFeeConsistency(unittest.TestCase):
    def setUp(self):
        self.params = Params()
        self.params.fund_fee = 0.01  # 1% fee for testing
        self.params.pension_fee = 0.005 # 0.5% pension fee
        self.params.annual_contribution = 10000
        self.params.years_accum = 10
        self.params.age_start = 40
        self.params.age_retire = 60
        self.params.years_transition = 5 # Satisfy validation (1-20)
        
        # Identity returns (no growth from market)
        self.eq_returns = np.zeros(self.params.years_accum + 1)
        self.bd_returns = np.zeros(self.params.years_accum + 1)

    def test_multiplicative_compounding_broker(self):
        """Test that broker accumulation matches manual multiplicative calculation."""
        scen = ScenarioBroker(self.params)
        pot = scen.accumulate(self.eq_returns, self.bd_returns)
        
        # Manual calculation:
        # Each year: val = (val * (1-0.01) + 10000)
        # Final year: val = val * (1-0.01)
        val = 0.0
        ann = 10000.0
        for _ in range(self.params.years_accum):
            val = (val * 0.99) + ann
            ann *= 1.02
        val *= 0.99
        
        self.assertAlmostEqual(pot.br_eq, val, places=2)

    def test_multiplicative_compounding_rurup(self):
        """Test that rurup accumulation matches manual multiplicative calculation."""
        scen = ScenarioRurupBroker(self.params)
        pot = scen.accumulate(self.eq_returns, self.bd_returns)
        
        # Manual calculation for rp:
        # Each year: rp = (rp * (1-0.01) * (1-0.005) + 10000)
        # Final year: rp = rp * (1-0.01) * (1-0.005)
        rp = 0.0
        ann = 10000.0
        for _ in range(self.params.years_accum):
            rp = (rp * 0.99 * 0.995) + ann
            ann *= 1.02
        rp *= 0.99 * 0.995
        
        self.assertAlmostEqual(pot.rurup, rp, places=2)

    def test_multiplicative_compounding_l3(self):
        """Test that L3 accumulation matches manual multiplicative calculation."""
        scen = ScenarioL3(self.params)
        pot = scen.accumulate(self.eq_returns, self.bd_returns)
        
        # Manual calculation for l3:
        # Each year: l3 = (l3 * (1-0.01) * (1-0.005) + 10000)
        # Final year: l3 = l3 * (1-0.01) * (1-0.005)
        l3 = 0.0
        ann = 10000.0
        for _ in range(self.params.years_accum):
            l3 = (l3 * 0.99 * 0.995) + ann
            ann *= 1.02
        l3 *= 0.99 * 0.995
        
        self.assertAlmostEqual(pot.l3_eq, l3, places=2)

    def test_decumulation_formula_consistency(self):
        """Test that decumulation use multiplicative compounding in all scenarios."""
        scenarios = [
            ScenarioBroker(self.params),
            ScenarioL3(self.params),
            ScenarioL3Broker(self.params),
            ScenarioEnhancedBroker(self.params),
            ScenarioEnhancedL3Broker(self.params)
        ]
        
        for scen in scenarios:
            self.params.glide_path_years = 0
            pot = Pot(br_eq=100000, br_eq_bs=100000)
            # 10% return, 1% fee -> should be 100000 * 1.10 * 0.99 = 108900
            # If additive: 100000 * (1 + 0.10 - 0.01) = 109000
            rand_returns = {"eq": 0.10, "bd": 0.05}
            _, new_pot = scen.decumulate_year(pot, 0, 0, 0, rand_returns)
            
            self.assertAlmostEqual(new_pot.br_eq, 108900, msg=f"Failed for {scen.__class__.__name__}")

if __name__ == '__main__':
    unittest.main()
