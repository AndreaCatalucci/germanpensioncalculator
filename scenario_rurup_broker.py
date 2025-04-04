from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw


class ScenarioRurupBroker(Scenario):
    """Rürup + Broker => invests refund in broker pot."""

    def accumulate(self, eq_returns=None, bd_returns=None) -> Pot:
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

            rp *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
            rp += ann_c
            # broker invests refund
            ref_ = ann_c * self.params.tax_working
            br *= 1 + eq_r - self.params.fund_fee
            br += ref_
            br_bs += ref_

            ann_c *= 1.02

        # Final year growth
        if eq_returns is None or len(eq_returns) <= self.params.years_accum:
            raise ValueError("Bootstrapped returns are required for final year")
        final_eq_r = eq_returns[self.params.years_accum]

        rp *= 1 + final_eq_r - self.params.fund_fee - self.params.pension_fee
        br *= 1 + final_eq_r - self.params.fund_fee

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
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        if current_year < self.params.glide_path_years:
            frac = 1.0 / self.params.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        needed_broker = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed_broker, self.params.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot
