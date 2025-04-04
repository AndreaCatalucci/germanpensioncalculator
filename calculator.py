from params import Params
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_l3 import ScenarioL3
from scenario_rurup_l3 import ScenarioRurupL3
from scenario_l3_broker import ScenarioL3Broker
from scenario_safe_spend import ScenarioSafeSpend
from scenario_safe_retire import ScenarioSafeRetire
from simulation import simulate_montecarlo, plot_boxplot


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    p = Params()

    # Create scenarios
    sA = ScenarioBroker(p)
    sB = ScenarioRurupBroker(p)
    sC = ScenarioL3(p)
    sD = ScenarioRurupL3(p)
    sE = ScenarioL3Broker(p)
    sF = ScenarioSafeSpend(p)  # Optimized for spending
    sG = ScenarioSafeRetire(p)  # Optimized for safety

    # simulate
    rA = simulate_montecarlo(sA)
    rB = simulate_montecarlo(sB)
    rC = simulate_montecarlo(sC)
    rD = simulate_montecarlo(sD)
    rE = simulate_montecarlo(sE)
    rF = simulate_montecarlo(sF)  # Simulate SafeSpend
    rG = simulate_montecarlo(sG)  # Simulate SafeRetire

    # Sort by p50pot
    combos = [
        ("Broker", rA),
        ("RurupBroker", rB),
        ("L3", rC),
        ("RurupL3", rD),
        ("L3Broker", rE),
        ("SafeSpend", rF),
        ("SafeRetire", rG),
    ]
    combos_sorted = sorted(combos, key=lambda x: x[1]["p50pot"], reverse=True)

    for name, res in combos_sorted:
        print(
            f"Scenario {name}: leftover p50={res['p50pot']:.0f}, run-out={res['prob_runout'] * 100:.1f}%, "
            f"p50 spend={res['p50']:,.0f}"
        )

    # box plot
    plot_boxplot([c[1] for c in combos_sorted], [c[0] for c in combos_sorted])
