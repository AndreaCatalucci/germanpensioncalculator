from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw


class ScenarioL3(Scenario):
    """L3 lumpsum half CG => eq->bd decum."""

    def accumulate(self, eq_returns=None, bd_returns=None) -> Pot:
        pot = Pot()
        val, bs = 0.0, 0.0
        c = self.params.annual_contribution
        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]

            val *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
            val += c
            bs += c
            c *= 1.02

        # Final year growth
        if eq_returns is None or len(eq_returns) <= self.params.years_accum:
            raise ValueError("Bootstrapped returns are required for final year")
        final_eq_r = eq_returns[self.params.years_accum]
        val *= 1 + final_eq_r - self.params.fund_fee - self.params.pension_fee
        pot.l3_eq = val
        pot.l3_eq_bs = bs
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

        withdrawn = withdraw(
            pot, max(0, needed_net - net_ann), self.params.cg_tax_normal
        )
        return withdrawn.net_withdrawn + net_ann, pot
