from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Type
from params import Params
from scenario_broker import ScenarioBroker
from scenario_enhanced_broker import ScenarioEnhancedBroker
from scenario_enhanced_l3_broker import ScenarioEnhancedL3Broker
from scenario_l3_broker import ScenarioL3Broker
from scenario_rurup_broker import ScenarioRurupBroker
from simulation import plot_boxplot, simulate_montecarlo

if TYPE_CHECKING:
    from scenario_base import Scenario


def run_scenarios(scenario_list: List[Tuple[str, Type[Scenario]]]) -> None:
    """
    Run simulations for a list of scenarios and display results.

    Args:
        scenario_list: List of tuples (name, scenario_class)
    """
    p = Params()

    # Create scenario instances and run simulations
    results = []
    for name, scenario_class in scenario_list:
        scenario = scenario_class(p)
        result = simulate_montecarlo(scenario)
        results.append((name, result))

    # Sort by p50pot (median leftover pot)
    results_sorted = sorted(results, key=lambda x: x[1]["p50pot"], reverse=True)

    # Print results
    for name, res in results_sorted:
        print(
            f"Scenario {name}: leftover p50={res['p50pot']:.0f}, run-out={res['prob_runout'] * 100:.1f}%, "
            f"p50 spend={res['p50']:,.0f}"
        )

    # Generate boxplot
    plot_boxplot([r[1] for r in results_sorted], [r[0] for r in results_sorted])


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    # Define scenarios to run
    scenarios: List[Tuple[str, Type[Scenario]]] = [
        ("Broker", ScenarioBroker),
        ("RurupBroker", ScenarioRurupBroker),
        # ("L3", ScenarioL3),
        # ("RurupL3", ScenarioRurupL3),
        ("L3Broker", ScenarioL3Broker),
        # ("SafeSpend", ScenarioSafeSpend),
        # ("SafeRetire", ScenarioSafeRetire),
        ("EnhancedBroker", ScenarioEnhancedBroker),
        ("EnhancedL3Broker", ScenarioEnhancedL3Broker),
    ]

    # Run scenarios
    run_scenarios(scenarios)
