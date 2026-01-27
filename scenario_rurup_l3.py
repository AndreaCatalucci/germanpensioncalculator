from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw

if TYPE_CHECKING:
    pass


class ScenarioRurupL3(Scenario):
    """Rürup + L3 lumpsum => eq->bond decum."""

    def accumulate(self, eq_returns: np.ndarray | None = None, bd_returns: np.ndarray | None = None) -> Pot:
        pot = Pot()
        rp, l3, l3_bs = 0.0, 0.0, 0.0
        c = self.params.annual_contribution
        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]
            # FIXED: Use multiplicative compounding for fees: (1 + return) * (1 - fees)
            growth_factor = (1 + eq_r) * (1 - self.params.fund_fee) * (1 - self.params.pension_fee)
            rp *= growth_factor
            rp += c
            # L3 invests the refund
            ref_ = c * self.params.tax_working
            l3 *= growth_factor
            l3 += ref_
            l3_bs += ref_
            c *= 1.02

        pot.rurup = rp
        pot.l3_eq = l3
        pot.l3_eq_bs = l3_bs
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict[str, float],
    ) -> tuple[float, Pot]:
        if current_year < self.params.glide_path_years:
            frac = 1.0 / self.params.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        needed = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed, self.params.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot
