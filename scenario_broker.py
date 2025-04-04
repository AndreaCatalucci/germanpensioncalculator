from scenario_base import Pot, Scenario, shift_equity_to_bonds, withdraw


class ScenarioBroker(Scenario):
    def accumulate(self, eq_returns=None, bd_returns=None) -> Pot:
        pot = Pot()
        eq_val = 0.0
        bs_val = 0.0
        ann_contr = self.params.annual_contribution

        for year in range(self.params.years_accum):
            # Use bootstrapped returns
            if eq_returns is None:
                raise ValueError("Bootstrapped returns are required")
            eq_r = eq_returns[year]
            eq_val *= 1 + eq_r - self.params.fund_fee
            eq_val += ann_contr
            bs_val += ann_contr
            ann_contr *= 1.02

        # Final year growth
        if eq_returns is None or len(eq_returns) <= self.params.years_accum:
            raise ValueError("Bootstrapped returns are required for final year")
        final_eq_r = eq_returns[self.params.years_accum]
        eq_val *= 1 + final_eq_r - self.params.fund_fee
        pot.br_eq = eq_val
        pot.br_eq_bs = bs_val
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

        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee

        withdrawn = withdraw(
            pot, max(0, needed_net - net_ann), self.params.cg_tax_normal
        )
        return withdrawn.net_withdrawn + net_ann, pot
