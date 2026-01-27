from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw, safe_multiply

if TYPE_CHECKING:
    pass


class ScenarioRurupBroker(Scenario):
    """Rürup + Broker => invests refund in broker pot."""

    def accumulate(self, eq_returns: np.ndarray | None = None, bd_returns: np.ndarray | None = None) -> Pot:
        pot = Pot()
        rp = 0.0
        br = 0.0
        br_bs = 0.0
        ann_c = self.params.annual_contribution
        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]
            # FIXED: Use multiplicative compounding for fees: (1 + return) * (1 - fees)
            growth_factor_rp = (1 + eq_r) * (1 - self.params.fund_fee) * (1 - self.params.pension_fee)
            rp *= growth_factor_rp
            rp += ann_c
            
            # broker invests refund
            ref_ = ann_c * self.params.tax_working
            growth_factor_br = (1 + eq_r) * (1 - self.params.fund_fee)
            br *= growth_factor_br
            br += ref_
            br_bs += ref_

            ann_c *= 1.02


        # store rp in eq, unify at retirement to net_ann
        pot.rurup = rp
        pot.br_eq = br
        pot.br_eq_bs = br_bs
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

        pot.br_eq = safe_multiply(pot.br_eq, 1 + rand_returns["eq"] - self.params.fund_fee)
        pot.br_bd = safe_multiply(pot.br_bd, 1 + rand_returns["bd"] - self.params.fund_fee)

        needed_broker = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed_broker, self.params.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot
