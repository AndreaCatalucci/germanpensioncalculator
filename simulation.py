from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from historical_returns import get_bond_returns, get_equity_returns
from scenario_base import Pot, present_value


def accumulate_initial_pots(p, eq_returns=None, bd_returns=None) -> Pot:
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
        eq_r = eq_returns[year]

        pot.rurup *= 1 + eq_r - p.fund_fee - p.pension_fee
        pot.br_eq *= 1 + eq_r - p.fund_fee
        pot.l3_eq *= 1 + eq_r - p.fund_fee - p.pension_fee

    # Final year growth
    if eq_returns is not None and len(eq_returns) > p.years_accum:
        final_eq_r = eq_returns[p.years_accum]
        pot.rurup *= 1 + final_eq_r - p.fund_fee - p.pension_fee
        pot.br_eq *= 1 + final_eq_r - p.fund_fee
        pot.l3_eq *= 1 + final_eq_r - p.fund_fee - p.pension_fee

    return pot


def collect_return_windows(equity_returns, bond_returns, window_size):
    """
    Collect all valid windows of returns from historical data.

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
        eq_mean = equity_returns.mean()
        eq_std = equity_returns.std()
        bd_mean = bond_returns.mean()
        bd_std = bond_returns.std()

        # Create a single synthetic window
        eq_window = np.random.normal(eq_mean, eq_std, window_size)
        bd_window = np.random.normal(bd_mean, bd_std, window_size)

        return [(eq_window, bd_window)]

    # Collect all valid windows
    windows = []
    for start_idx in range(len(common_years) - window_size + 1):
        years = common_years[start_idx : start_idx + window_size]
        eq_window = equity_returns.loc[years].values
        bd_window = bond_returns.loc[years].values
        windows.append((eq_window, bd_window))

    return windows


def sample_from_windows(windows, num_years):
    """
    Sample returns for a specified number of years by randomly selecting from windows.

    Args:
        windows: List of return windows (each window is a tuple of equity and bond returns)
        num_years: Number of years to sample

    Returns:
        tuple: (equity_returns, bond_returns) arrays for the sampled years
    """
    eq_returns = []
    bd_returns = []

    # Sample random windows until we have enough years
    while len(eq_returns) < num_years:
        # Randomly select a window
        window_idx = np.random.randint(0, len(windows))
        eq_window, bd_window = windows[window_idx]

        # Add returns from this window
        eq_returns.extend(eq_window)
        bd_returns.extend(bd_window)

    # Trim to the exact number of years needed
    return np.array(eq_returns[:num_years]), np.array(bd_returns[:num_years])


def simulate_montecarlo(scenario):
    p = scenario.params

    # Load historical returns data
    equity_returns = get_equity_returns(
        series_id=p.equity_series_id,
        start_year=p.data_start_year,
    )
    bond_returns = get_bond_returns(
        series_id=p.bond_series_id, start_year=p.data_start_year
    )

    # Collect windows of returns
    windows = collect_return_windows(
        equity_returns, bond_returns, p.bootstrap_period_length
    )

    # Sample returns for accumulation phase
    accumulation_eq_returns, accumulation_bd_returns = sample_from_windows(
        windows, p.years_accum + 1
    )

    init_pot = accumulate_initial_pots(
        p, accumulation_eq_returns, accumulation_bd_returns
    )
    scen_pot = scenario.accumulate(accumulation_eq_returns, accumulation_bd_returns)
    combined_pot = Pot(
        rurup=init_pot.rurup + scen_pot.rurup,
        l3_eq=init_pot.l3_eq + scen_pot.l3_eq,
        l3_eq_bs=init_pot.l3_eq_bs + scen_pot.l3_eq_bs,
        br_eq=init_pot.br_eq + scen_pot.br_eq,
        br_eq_bs=init_pot.br_eq_bs + scen_pot.br_eq_bs,
        br_bd=init_pot.br_bd + scen_pot.br_bd,
        br_bd_bs=init_pot.br_bd_bs + scen_pot.br_bd_bs,
    )
    
    # Import here to avoid circular imports
    from lifetime import sample_lifetime_from67
    
    lifetimes = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire
    results = []
    leftover_pots = []
    outcount = 0

    for i in range(p.num_sims):
        sim_pot = replace(combined_pot)  # copy
        total_spend = 0.0
        ran_out = False
        spend = p.desired_spend

        # Sample returns for transition phase
        transition_eq_returns, transition_bd_returns = sample_from_windows(
            windows, p.years_transition
        )

        for year in range(p.years_transition):
            eq_r = transition_eq_returns[year]
            bd_r = transition_bd_returns[year]
            scenario.transition_year(sim_pot, year, eq_r, bd_r)

        net_ann = scenario.params.public_pension
        if sim_pot.rurup > 0:
            gross_ann = sim_pot.rurup * p.ruerup_ann_rate
            net_ann += gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
            sim_pot.rurup = 0

        # Sample returns for retirement phase
        max_lifetime = int(np.ceil(lifetimes[i]))
        retirement_eq_returns, retirement_bd_returns = sample_from_windows(
            windows, max_lifetime
        )

        for t in range(lifetimes[i]):
            eq_r = retirement_eq_returns[t]
            bd_r = retirement_bd_returns[t]

            net_wd, sim_pot = scenario.decumulate_year(
                sim_pot, t, net_ann, spend, rand_returns={"eq": eq_r, "bd": bd_r}
            )
            total_spend += present_value(net_wd, t, p.discount_rate)
            if net_wd < spend:
                ran_out = True
                break
            infl = np.random.normal(p.inflation_mean, p.inflation_std)
            spend *= 1 + infl

        leftover_pots.append(sim_pot.leftover())
        results.append(total_spend)
        if ran_out:
            outcount += 1

    arr = np.array(results)
    return {
        "prob_runout": outcount / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p50pot": np.percentile(leftover_pots, 50),
        "all_spend": arr,
        "all_pots": np.array(leftover_pots),
    }


def plot_boxplot(data_list, labels):
    all_data = [d["all_spend"] for d in data_list]
    plt.figure(figsize=(7, 4))
    plt.boxplot(all_data, tick_labels=labels)
    plt.title("Boxplot of Discounted Spending (Sorted by leftover pot)")
    plt.ylabel("NPV of Spending")
    plt.grid(True)
    plt.show()
