from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw

if TYPE_CHECKING:
    pass


class ScenarioL3Broker(Scenario):
    """50% L3, 50% Broker => unify them at retirement so eq->bd decum."""

    def accumulate(self, eq_returns: np.ndarray | None = None, bd_returns: np.ndarray | None = None) -> Pot:
        pot = Pot()
        l3_eq = 0.0
        l3_eq_bs = 0.0
        br_eq, br_bd = 0.0, 0.0
        br_eq_bs, br_bd_bs = 0.0, 0.0
        c = self.params.annual_contribution
        proportion_l3 = 0.4

        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None or bd_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]
            bd_r = bd_returns[year]

            l3_eq *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
            br_eq *= 1 + eq_r - self.params.fund_fee
            br_bd *= 1 + bd_r - self.params.fund_fee

            c_l3 = c * proportion_l3
            c_br = c * (1 - proportion_l3)
            l3_eq += c_l3
            l3_eq_bs += c_l3
            br_eq += c_br
            br_eq_bs += c_br
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
        rand_returns: dict[str, float],
    ) -> tuple[float, Pot]:
        age_now = self.params.age_retire + current_year
        if age_now >= 70 and (age_now - 70) < self.params.glide_path_years:
            frac = 1.0 / self.params.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"] - self.params.fund_fee
        pot.br_bd *= 1 + rand_returns["bd"] - self.params.fund_fee

        # partial withdrawal
        withdrawn = withdraw(
            pot, max(0, needed_net - net_ann), self.params.cg_tax_normal
        )
        return withdrawn.net_withdrawn + net_ann, pot
