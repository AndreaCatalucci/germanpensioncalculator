from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING, cast, TypedDict, List
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from historical_returns import get_bond_returns, get_equity_returns
from params import Params
from scenario_base import Pot, present_value

if TYPE_CHECKING:
    from scenario_base import Scenario


class SimulationResult(TypedDict):
    prob_runout: float
    p10: float
    p50: float
    p90: float
    p50pot: float
    all_spend: npt.NDArray[np.float64]
    all_pots: npt.NDArray[np.float64]
    median_trajectory: List[Pot]
    p10_trajectory: List[Pot]
    p90_trajectory: List[Pot]
    p50_income_split: Dict[str, npt.NDArray[np.float64]]


def accumulate_initial_pots(
    p: Params,
    eq_returns: npt.NDArray[np.float64] | None = None,
    bd_returns: npt.NDArray[np.float64] | None = None,
) -> Pot:
    """Grow the user's initial pots from age_start..age_retire."""
    pot = Pot(
        rurup=p.initial_rurup,
        br_eq=p.initial_broker,
        br_eq_bs=p.initial_broker_bs,
        l3_eq=p.initial_l3,
        l3_eq_bs=p.initial_l3_bs,
    )
    for year in range(p.years_accum):
        # Use bootstrapped returns
        if eq_returns is None:
            raise ValueError("Bootstrapped returns are required")
        eq_r = float(eq_returns[year])

        # FIXED: Use multiplicative compounding for fees: (1 + return) * (1 - fees)
        # Add numerical stability checks
        growth_factor_rurup = max(0.01, (1 + eq_r) * (1 - p.fund_fee) * (1 - p.pension_fee))
        growth_factor_br = max(0.01, (1 + eq_r) * (1 - p.fund_fee))
        growth_factor_l3 = max(0.01, (1 + eq_r) * (1 - p.fund_fee) * (1 - p.pension_fee))
        
        pot.rurup *= growth_factor_rurup
        pot.br_eq *= growth_factor_br
        pot.l3_eq *= growth_factor_l3


    return pot


def collect_return_windows(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    window_size: int,
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Collect all valid overlapping windows of returns from historical data.
    
    FIXED: Now creates overlapping windows for better statistical coverage.

    Args:
        equity_returns: Series of historical equity returns
        bond_returns: Series of historical bond returns
        window_size: Size of each window (e.g., 10 years)

    Returns:
        list of tuples: Each tuple contains (equity_window, bond_window)
    """
    # Ensure we have the same years for both series
    common_years = sorted(
        set(equity_returns.index).intersection(set(bond_returns.index))
    )

    if len(common_years) < window_size:
        print(
            f"Warning: Not enough data for {window_size}-year windows. Using synthetic data."
        )
        # Generate synthetic data based on historical means and standard deviations
        eq_mean = equity_returns.mean() if len(equity_returns) > 0 else 0.08
        eq_std = equity_returns.std() if len(equity_returns) > 0 else 0.18
        bd_mean = bond_returns.mean() if len(bond_returns) > 0 else 0.04
        bd_std = bond_returns.std() if len(bond_returns) > 0 else 0.05

        # Create multiple synthetic windows for better coverage
        num_synthetic_windows = max(10, window_size)
        windows = []
        for _ in range(num_synthetic_windows):
            eq_window = np.clip(np.random.normal(eq_mean, eq_std, window_size), -0.5, 0.9)
            bd_window = np.clip(np.random.normal(bd_mean, bd_std, window_size), -0.2, 0.4)
            windows.append((eq_window, bd_window))

        return windows

    # Collect all valid overlapping windows
    windows = []
    for start_idx in range(len(common_years) - window_size + 1):
        years = common_years[start_idx : start_idx + window_size]
        eq_window = cast(npt.NDArray[np.float64], equity_returns.loc[years].values)
        bd_window = cast(npt.NDArray[np.float64], bond_returns.loc[years].values)
        windows.append((eq_window, bd_window))

    return windows


def sample_from_windows(
    windows: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    num_years: int,
    equity_safety_margin: float = 0,
    bond_safety_margin: float = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Sample returns for a specified number of years by randomly selecting individual years from windows.
    
    FIXED: Now samples individual years instead of sequential windows to avoid bias.

    Args:
        windows: List of return windows (each window is a tuple of equity and bond returns)
        num_years: Number of years to sample
        equity_safety_margin: Safety margin to apply to equity returns (negative for conservative)
        bond_safety_margin: Safety margin to apply to bond returns (negative for conservative)

    Returns:
        tuple: (equity_returns, bond_returns) arrays for the sampled years
    """
    if not windows:
        raise ValueError("No windows available for sampling")

    eq_returns: list[float] = []
    bd_returns: list[float] = []

    # Create a flat list of all individual year returns from all windows
    all_eq_returns: list[float] = []
    all_bd_returns: list[float] = []

    for eq_window, bd_window in windows:
        all_eq_returns.extend(cast(List[float], eq_window.flatten().tolist()))
        all_bd_returns.extend(cast(List[float], bd_window.flatten().tolist()))
    
    if len(all_eq_returns) == 0:
        raise ValueError("No returns available in windows")

    # Sample individual years randomly with replacement and apply safety margins
    for _ in range(num_years):
        idx = np.random.randint(0, len(all_eq_returns))
        eq_returns.append(all_eq_returns[idx] + equity_safety_margin)
        bd_returns.append(all_bd_returns[idx] + bond_safety_margin)

    return np.array(eq_returns), np.array(bd_returns)


def simulate_montecarlo(scenario: Scenario) -> SimulationResult:
    p = scenario.params

    # Load historical returns data
    equity_returns = get_equity_returns(
        series_id=p.equity_series_id, start_year=p.data_start_year
    )
    bond_returns = get_bond_returns(
        series_id=p.bond_series_id, start_year=p.data_start_year
    )

    # Collect windows of returns
    windows = collect_return_windows(
        equity_returns, bond_returns, p.bootstrap_period_length
    )

    # Import here to avoid circular imports
    from lifetime import sample_lifetime_from67
    
    lifetimes = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire
    results = []
    leftover_pots = []
    trajectories = []
    income_splits = []
    outcount = 0

    for i in range(p.num_sims):
        # Sample returns for accumulation phase
        # FIXED: Each run now gets its own unique accumulation path
        acc_eq_r, acc_bd_r = sample_from_windows(
            windows, p.years_accum + 1, getattr(p, 'equity_safety_margin', 0), getattr(p, 'bond_safety_margin', 0)
        )
        
        init_pot = accumulate_initial_pots(p, acc_eq_r, acc_bd_r)
        scen_pot = scenario.accumulate(acc_eq_r, acc_bd_r)
        
        sim_pot = Pot(
            rurup=float(init_pot.rurup + scen_pot.rurup),
            l3_eq=float(init_pot.l3_eq + scen_pot.l3_eq),
            l3_eq_bs=float(init_pot.l3_eq_bs + scen_pot.l3_eq_bs),
            br_eq=float(init_pot.br_eq + scen_pot.br_eq),
            br_eq_bs=float(init_pot.br_eq_bs + scen_pot.br_eq_bs),
            br_bd=float(init_pot.br_bd + scen_pot.br_bd),
            br_bd_bs=float(init_pot.br_bd_bs + scen_pot.br_bd_bs),
        )
        
        current_trajectory = [replace(sim_pot)]
        current_income_split = {
            "pension": [],
            "rurup": [],
            "broker_net": []
        }
        total_spend = 0.0
        ran_out = False
        spend = p.desired_spend

        # Sample returns for transition phase
        transition_eq_returns, transition_bd_returns = sample_from_windows(
            windows, p.years_transition, getattr(p, 'equity_safety_margin', 0), getattr(p, 'bond_safety_margin', 0)
        )

        for year in range(p.years_transition):
            eq_r = float(transition_eq_returns[year])
            bd_r = float(transition_bd_returns[year])
            scenario.transition_year(sim_pot, year, eq_r, bd_r)

        net_ann = scenario.params.public_pension
        if sim_pot.rurup > 0:
            # Add numerical stability check for annuity rate
            safe_ann_rate = max(0.001, min(1.0, p.ruerup_ann_rate))  # Bound between 0.1% and 100%
            gross_ann = sim_pot.rurup * safe_ann_rate
            
            # FIXED: Accurate Rürup taxation logic
            # net = gross * (1 - (taxable_share * marginal_tax_rate))
            # Also subtract distribution fee
            tax_rate = getattr(p, 'tax_retirement', 0.30)
            tax_share = getattr(p, 'ruerup_tax_share', 1.0)
            dist_fee = getattr(p, 'ruerup_dist_fee', 0.0)
            
            tax_factor = 1.0 - (tax_share * tax_rate)
            rurup_annuity = gross_ann * tax_factor * (1 - dist_fee)
            net_ann += rurup_annuity
            sim_pot.rurup = 0
        else:
            rurup_annuity = 0.0

        # Sample returns for retirement phase
        max_lifetime = int(np.ceil(lifetimes[i]))
        retirement_eq_returns, retirement_bd_returns = sample_from_windows(
            windows, max_lifetime, getattr(p, 'equity_safety_margin', 0), getattr(p, 'bond_safety_margin', 0)
        )

        for t in range(lifetimes[i]):
            eq_r = float(retirement_eq_returns[t])
            bd_r = float(retirement_bd_returns[t])

            net_wd, sim_pot = scenario.decumulate_year(
                sim_pot, t, net_ann, spend, rand_returns={"eq": eq_r, "bd": bd_r}
            )
            # Add numerical stability check for present value calculation
            safe_discount_rate = max(0.001, p.discount_rate)  # Minimum 0.1% discount rate
            total_spend += present_value(net_wd, t, safe_discount_rate)
            
            # Capture income breakdown (before inflation adjustment)
            # scenario.decumulate_year returns (withdrawn_from_broker + net_ann)
            # So broker_withdrawal = max(0, net_wd - net_ann)
            broker_wd = max(0.0, net_wd - net_ann)
            current_income_split["pension"].append(p.public_pension)
            current_income_split["rurup"].append(rurup_annuity)
            current_income_split["broker_net"].append(broker_wd)
            current_trajectory.append(replace(sim_pot))

            if net_wd < spend:
                ran_out = True
                # Fill remaining years of lifetime with zeros/current state if ran out
                remaining = int(np.ceil(lifetimes[i])) - t - 1
                for _ in range(remaining):
                    current_income_split["pension"].append(0.0)
                    current_income_split["rurup"].append(0.0)
                    current_income_split["broker_net"].append(0.0)
                    current_trajectory.append(replace(sim_pot))
                break
            infl = np.random.normal(p.inflation_mean, p.inflation_std)
            spend *= 1 + infl

        leftover_pots.append(sim_pot.leftover())
        results.append(total_spend)
        trajectories.append(current_trajectory)
        
        # Ensure income split arrays are numpy for later processing
        income_splits.append({
            k: np.array(v) for k, v in current_income_split.items()
        })
        
        if ran_out:
            outcount += 1

    arr = np.array(results)
    
    # Find the indices for p10, p50, p90 based on total_spend
    # This allows us to show representative trajectories for different outcomes
    p10_val = np.percentile(arr, 10)
    p50_val = np.percentile(arr, 50)
    p90_val = np.percentile(arr, 90)
    
    p10_idx = int(np.argmin(np.abs(arr - p10_val)))
    median_idx = int(np.argmin(np.abs(arr - p50_val)))
    p90_idx = int(np.argmin(np.abs(arr - p90_val)))
    
    return {
        "prob_runout": outcount / p.num_sims,
        "p10": p10_val,
        "p50": p50_val,
        "p90": p90_val,
        "p50pot": np.percentile(leftover_pots, 50),
        "all_spend": arr,
        "all_pots": np.array(leftover_pots),
        "median_trajectory": trajectories[median_idx],
        "p10_trajectory": trajectories[p10_idx],
        "p90_trajectory": trajectories[p90_idx],
        "p50_income_split": income_splits[median_idx]
    }


def plot_boxplot(data_list: list[SimulationResult], labels: list[str]) -> None:
    all_data: list[npt.NDArray[np.float64]] = [
        d["all_spend"] for d in data_list
    ]
    plt.figure(figsize=(7, 4))
    plt.boxplot(all_data, tick_labels=labels)
    plt.title("Boxplot of Discounted Spending (Sorted by leftover pot)")
    plt.ylabel("NPV of Spending")
    plt.grid(True)
    plt.show()
