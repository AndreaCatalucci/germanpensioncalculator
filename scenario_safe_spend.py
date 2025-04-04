from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw


class ScenarioSafeSpend(Scenario):
    """Hybrid strategy optimized for minimizing run-out risk while maximizing spending."""

    def __init__(self, p):
        super().__init__(p)
        # Adjust the glide path to be more conservative
        self.glide_path_years = 15  # Shorter glide path (more conservative)
        self.early_shift_years = 5  # Start shifting to bonds 5 years before retirement
        self.l3_proportion = 0.5  # Equal split between L3 and Broker
        self.initial_bond_allocation = 0.2  # Start with 20% in bonds
        self.floor_withdrawal_pct = 0.035  # Minimum withdrawal rate (3.5%)
        self.ceiling_withdrawal_pct = 0.055  # Maximum withdrawal rate (5.5%)
        self.target_withdrawal_pct = 0.045  # Target withdrawal rate (4.5%)
        self.market_sensitivity = (
            0.5  # How much to adjust based on market performance (0-1)
        )

    def accumulate(self, eq_returns=None, bd_returns=None) -> Pot:
        pot = Pot()
        l3_eq = 0.0
        l3_eq_bs = 0.0
        br_eq, br_bd = 0.0, 0.0
        br_eq_bs, br_bd_bs = 0.0, 0.0
        c = self.params.annual_contribution

        # Initial bond allocation
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
                    1.0 - self.initial_bond_allocation
                ) / self.early_shift_years

            # L3 portion (all equity)
            l3_eq *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee

            # Broker portion (mix of equity and bonds)
            br_eq *= 1 + eq_r - self.params.fund_fee
            br_bd *= 1 + bd_r - self.params.fund_fee

            # Allocate new contributions
            c_l3 = c * self.l3_proportion
            c_br = c * (1 - self.l3_proportion)

            # Add to L3
            l3_eq += c_l3
            l3_eq_bs += c_l3

            # Add to Broker with bond allocation
            br_eq += c_br * (1 - bond_allocation)
            br_eq_bs += c_br * (1 - bond_allocation)
            br_bd += c_br * bond_allocation
            br_bd_bs += c_br * bond_allocation

            c *= 1.02

        # Final year growth
        if eq_returns is None or len(eq_returns) <= self.params.years_accum or bd_returns is None or len(bd_returns) <= self.params.years_accum:
            raise ValueError("Bootstrapped returns are required for final year")
        final_eq_r = eq_returns[self.params.years_accum]
        final_bd_r = bd_returns[self.params.years_accum]

        l3_eq *= 1 + final_eq_r - self.params.fund_fee - self.params.pension_fee
        br_eq *= 1 + final_eq_r - self.params.fund_fee
        br_bd *= 1 + final_bd_r - self.params.fund_fee

        pot.l3_eq = l3_eq
        pot.l3_eq_bs = l3_eq_bs
        pot.br_eq, pot.br_bd = br_eq, br_bd
        pot.br_eq_bs, pot.br_bd_bs = br_eq_bs, br_bd_bs
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        # More aggressive shift to bonds in early retirement
        if current_year < self.glide_path_years:
            frac = 1.0 / self.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        # Grow the portfolio
        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee

        # Dynamic withdrawal strategy
        total_portfolio = pot.br_eq + pot.br_bd + pot.l3_eq

        # Base withdrawal on portfolio value and target rate
        base_withdrawal = total_portfolio * self.target_withdrawal_pct

        # Adjust based on recent market performance
        market_adjustment = 1.0
        # Use a dynamic approach based on historical average
        # We'll calculate the reference return from the historical data
        # This would typically be done once at initialization, but for simplicity
        # we'll use a fixed value here
        reference_return = 0.08  # Historical average equity return
        if eq_r > reference_return:
            # Good market year - can withdraw a bit more
            market_adjustment = (
                1.0 + (eq_r - reference_return) * self.market_sensitivity
            )
        elif eq_r < reference_return:
            # Bad market year - withdraw a bit less
            market_adjustment = (
                1.0 - (reference_return - eq_r) * self.market_sensitivity
            )

        # Apply market adjustment
        adjusted_withdrawal = base_withdrawal * market_adjustment

        # Apply floor and ceiling
        floor_withdrawal = total_portfolio * self.floor_withdrawal_pct
        ceiling_withdrawal = total_portfolio * self.ceiling_withdrawal_pct

        final_withdrawal = max(
            min(adjusted_withdrawal, ceiling_withdrawal), floor_withdrawal
        )

        # Ensure we don't withdraw more than needed
        final_withdrawal = min(final_withdrawal, needed_net - net_ann)

        # If we need more than the final withdrawal, we'll withdraw that amount
        if final_withdrawal < needed_net - net_ann:
            final_withdrawal = needed_net - net_ann

        # Withdraw from the portfolio
        withdrawn = withdraw(pot, final_withdrawal, self.params.cg_tax_normal)

        return withdrawn.net_withdrawn + net_ann, pot
