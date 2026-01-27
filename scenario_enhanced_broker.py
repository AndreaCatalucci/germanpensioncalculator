from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw

if TYPE_CHECKING:
    from params import Params


class ScenarioEnhancedBroker(Scenario):
    """Enhanced Broker strategy optimized for 0% run-out and maximum spending."""

    def __init__(self, p: Params) -> None:
        super().__init__(p)
        # Strategy parameters
        self.early_shift_years = 5  # Start shifting to bonds 5 years before retirement
        self.glide_path_years = 15  # Shorter glide path than original Broker
        self.glide_path_start_age = 70  # Delay glide path until age 70
        self.initial_bond_allocation = 0.10  # Start with 10% bonds
        self.pre_retirement_bond_target = 0.20  # Target 20% bonds before retirement
        self.base_withdrawal_rate = 0.048  # 4.8% base withdrawal rate
        self.market_sensitivity = 0.15  # Adjust withdrawals by up to 15% based on market
        self.safety_buffer = 0.05  # 5% safety buffer

    def accumulate(self, eq_returns: np.ndarray | None = None, bd_returns: np.ndarray | None = None) -> Pot:
        pot = Pot()
        eq_val = 0.0
        bd_val = 0.0
        bs_eq = 0.0
        bs_bd = 0.0
        ann_contr = self.params.annual_contribution
        bond_allocation = self.initial_bond_allocation

        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None or bd_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]
            bd_r = bd_returns[year]

            # Increase bond allocation in the last few years before transition
            if year >= self.params.years_accum - self.early_shift_years:
                bond_allocation += (
                    self.pre_retirement_bond_target - self.initial_bond_allocation
                ) / self.early_shift_years

            # Grow existing investments
            eq_val *= 1 + eq_r - self.params.fund_fee
            bd_val *= 1 + bd_r - self.params.fund_fee

            # Allocate new contributions
            eq_contribution = ann_contr * (1 - bond_allocation)
            bd_contribution = ann_contr * bond_allocation

            eq_val += eq_contribution
            bd_val += bd_contribution
            bs_eq += eq_contribution
            bs_bd += bd_contribution

            ann_contr *= 1.02  # Increase contribution by 2% per year

        # Final year growth
        if eq_returns is None or len(eq_returns) <= self.params.years_accum or bd_returns is None or len(bd_returns) <= self.params.years_accum:
            raise ValueError("Bootstrapped returns are required for final year")
        final_eq_r = eq_returns[self.params.years_accum]
        final_bd_r = bd_returns[self.params.years_accum]

        eq_val *= 1 + final_eq_r - self.params.fund_fee
        bd_val *= 1 + final_bd_r - self.params.fund_fee

        pot.br_eq = eq_val
        pot.br_bd = bd_val
        pot.br_eq_bs = bs_eq
        pot.br_bd_bs = bs_bd
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict[str, float],
    ) -> tuple[float, Pot]:
        # Calculate current age
        age_now = self.params.age_retire + current_year

        # Start glide path at specified age
        if age_now >= self.glide_path_start_age and (
            age_now - self.glide_path_start_age
        ) < self.glide_path_years:
            frac = 1.0 / self.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        # Grow the portfolio
        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee

        # Dynamic withdrawal strategy
        total_portfolio = pot.br_eq + pot.br_bd
        available_portfolio = total_portfolio * (1 - self.safety_buffer)

        # Base withdrawal on portfolio value and target rate
        base_withdrawal = available_portfolio * self.base_withdrawal_rate

        # Adjust based on recent market performance
        market_adjustment = 1.0
        reference_return = 0.08  # Historical average equity return
        if eq_r > reference_return:
            # Good market year - can withdraw a bit more
            market_adjustment = 1.0 + (eq_r - reference_return) * self.market_sensitivity
        elif eq_r < reference_return:
            # Bad market year - withdraw a bit less
            market_adjustment = 1.0 - (reference_return - eq_r) * self.market_sensitivity

        # Apply market adjustment
        adjusted_withdrawal = base_withdrawal * market_adjustment

        # Ensure we don't withdraw more than needed
        final_withdrawal = min(adjusted_withdrawal, needed_net - net_ann)

        # If we need more than the adjusted withdrawal, we'll withdraw that amount
        if final_withdrawal < needed_net - net_ann:
            final_withdrawal = needed_net - net_ann

        # Withdraw from the portfolio
        withdrawn = withdraw(pot, final_withdrawal, self.params.cg_tax_normal)

        return withdrawn.net_withdrawn + net_ann, pot
