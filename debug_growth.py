from params import Params
from scenario_rurup_broker import ScenarioRurupBroker
import numpy as np

p = Params()
p.annual_contribution = 20000
p.years_accum = 24
p.initial_rurup = 20000
p.initial_broker = 100000
p.tax_working = 0.4431
p.fund_fee = 0.002
p.pension_fee = 0.003

scenario = ScenarioRurupBroker(p)
# Use a constant 7% return
eq_returns = np.full(30, 0.07)
bd_returns = np.full(30, 0.03)

print("Starting accumulation...")
pot = scenario.accumulate(eq_returns=eq_returns)
print(f"At age 62: Rurup={pot.rurup:,.0f}, Broker={pot.br_eq:,.0f}")

# Transition for 5 years
for y in range(5):
    pot = scenario.transition_year(pot, y, 0.07, 0.03)
print(f"At age 67: Rurup={pot.rurup:,.0f}, Broker Eq={pot.br_eq:,.0f}, Broker Bd={pot.br_bd:,.0f}")

# Payout
safe_ann_rate = 0.028
tax_rate = 0.30
tax_share = 1.0
dist_fee = 0.015

gross_ann = pot.rurup * safe_ann_rate
tax_factor = 1.0 - (tax_share * tax_rate)
net_ann = gross_ann * tax_factor * (1 - dist_fee)
print(f"Annual Net Annuity: {net_ann:,.0f}")
