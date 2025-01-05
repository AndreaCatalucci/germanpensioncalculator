import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from params import Params
from lifetime import sample_lifetime_from67


@dataclass
class Pot:
    rurup: float = 0.0

    br_eq: float = 0.0
    br_eq_bs: float = 0.0
    br_bd: float = 0.0
    br_bd_bs: float = 0.0

    l3_eq: float = 0.0
    l3_eq_bs: float = 0.0

    def leftover(self):
        return self.br_eq + self.br_bd + self.l3_eq


@dataclass
class Withdrawal:
    gross_withdrawn: float
    net_withdrawn: float


def present_value(cf, year, discount_rate):
    return cf / ((1 + discount_rate) ** year)


def withdraw(pot: Pot, net_needed, cg_tax) -> Withdrawal:
    total_pot = pot.br_bd + pot.br_eq + pot.l3_eq
    if total_pot <= 0:
        return Withdrawal(0.0, 0.0)

    net_withdrawn = 0
    gross_withdrawn = 0
    still_needed = net_needed

    # BOND FIRST
    if pot.br_bd > 0:
        net_bd = pot.br_bd - (pot.br_bd - pot.br_bd_bs) * cg_tax
        portion = min(still_needed, net_bd)
        if portion > 0:
            frac = portion / net_bd
            gross_bd_withdrawal = frac * pot.br_bd
            pot.br_bd_bs = pot.br_bd_bs * (1 - frac)
            pot.br_bd -= gross_bd_withdrawal
            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # EQUITY NEXT
    if still_needed > 0 and pot.br_eq > 0:
        net_eq = pot.br_eq - (pot.br_eq - pot.br_eq_bs) * cg_tax
        portion2 = min(still_needed, net_eq)
        if portion2 > 0:
            frac2 = portion2 / net_eq
            gross_eq_withdrawal = frac2 * pot.br_eq
            pot.br_eq_bs = pot.br_eq_bs * (1 - frac2)
            pot.br_eq -= gross_eq_withdrawal
            net_withdrawn += portion2
            gross_withdrawn += gross_eq_withdrawal
            still_needed -= portion2

    return Withdrawal(gross_withdrawn, net_withdrawn)


def shift_equity_to_bonds(pot: Pot, fraction, cgt):
    moved = pot.br_eq * fraction
    bs_moved = pot.br_eq_bs * fraction
    pot.br_eq -= moved
    pot.br_eq_bs -= bs_moved
    gains = moved - bs_moved
    tax_amount = gains * cgt
    moved_net = moved - tax_amount
    pot.br_bd += moved_net
    pot.br_bd_bs += moved_net


# --------------------------------------------------------
# 4. BASE SCENARIO
# --------------------------------------------------------
class Scenario:

    def __init__(self, p: Params):
        self.params = p

    def accumulate(self) -> Pot:
        raise NotImplementedError

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        raise NotImplementedError

    def transition_year(
        self, pot: Pot, current_year: int, eq_r: float, bd_r: float
    ) -> Pot:
        pot.rurup *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.l3_eq *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee
        if pot.l3_eq > 0:
            fraction = 1. / self.params.years_transition
            moved = fraction * pot.l3_eq
            fees = moved * self.params.ruerup_dist_fee
            gains = max(0, moved - pot.l3_eq_bs * fraction - fees)
            tax = gains * self.params.cg_tax_half
            net_l3 = max(0, moved - tax - fees)
            pot.br_bd += net_l3
            pot.br_bd_bs += net_l3
            pot.l3_eq -= moved
            pot.l3_eq_bs -= pot.l3_eq_bs * fraction


# --------------------------------------------------------
# 5. ACCUMULATE INITIAL POTS
# --------------------------------------------------------
def accumulate_initial_pots(p: Params) -> Pot:
    """Grow the user's initial pots from age_start..age_retire."""
    pot = Pot(
        rurup=p.initial_rurup,
        br_eq=p.initial_broker,
        br_eq_bs=p.initial_broker_bs,
        l3_eq=p.initial_l3,
        l3_eq_bs=p.initial_l3_bs,
    )
    for _ in range(p.years_accum):
        pot.rurup *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        pot.br_eq *= 1 + p.equity_mean - p.fund_fee
        pot.l3_eq *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
    return pot


class ScenarioBroker(Scenario):
    def accumulate(self) -> Pot:
        pot = Pot()
        eq_val = 0.0
        bs_val = 0.0
        ann_contr = self.params.annual_contribution

        for _ in range(self.params.years_accum):
            eq_val *= 1 + self.params.equity_mean - self.params.fund_fee
            eq_val += ann_contr
            bs_val += ann_contr
            ann_contr *= 1.02

        eq_val *= 1 + self.params.equity_mean - self.params.fund_fee
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
            shift_equity_to_bonds(
                pot,
                frac,
                self.params.cg_tax_normal,
            )

        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee

        withdrawn = withdraw(
            pot,
            max(0, needed_net - net_ann),
            self.params.cg_tax_normal,
        )
        return withdrawn.net_withdrawn + net_ann, pot


class ScenarioRurupBroker(Scenario):
    """Rürup + Broker => invests refund in broker pot."""

    def accumulate(self) -> Pot:
        pot = Pot()
        rp = 0.0
        br = 0.0
        br_bs = 0.0
        ann_c = self.params.annual_contribution
        for _ in range(self.params.years_accum):
            rp *= (
                1
                + self.params.equity_mean
                - self.params.fund_fee
                - self.params.pension_fee
            )
            rp += ann_c
            # broker invests refund
            ref_ = ann_c * self.params.tax_working
            br *= 1 + self.params.equity_mean - self.params.fund_fee
            br += ref_
            br_bs += ref_

            ann_c *= 1.02

        rp *= (
            1 + self.params.equity_mean - self.params.fund_fee - self.params.pension_fee
        )
        br *= 1 + self.params.equity_mean - self.params.fund_fee

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


class ScenarioL3(Scenario):
    """L3 lumpsum half CG => eq->bd decum."""

    def accumulate(self) -> Pot:
        pot = Pot()
        val, bs = 0.0, 0.0
        c = self.params.annual_contribution
        for _ in range(self.params.years_accum):
            val *= (
                1
                + self.params.equity_mean
                - self.params.fund_fee
                - self.params.pension_fee
            )
            val += c
            bs += c
            c *= 1.02
        val *= (
            1 + self.params.equity_mean - self.params.fund_fee - self.params.pension_fee
        )
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
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            shift_equity_to_bonds(pot, frac, p.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        withdrawn = withdraw(
            pot,
            max(0, needed_net - net_ann),
            self.params.cg_tax_normal,
        )
        return withdrawn.net_withdrawn + net_ann, pot


class ScenarioRurupL3(Scenario):
    """Rürup + L3 lumpsum => eq->bond decum."""

    def accumulate(self) -> Pot:
        pot = Pot()
        rp, l3, l3_bs = 0.0, 0.0, 0.0
        c = self.params.annual_contribution
        for _ in range(self.params.years_accum):
            rp *= (
                1
                + self.params.equity_mean
                - self.params.fund_fee
                - self.params.pension_fee
            )
            rp += c
            # L3 invests the refund
            ref_ = c * self.params.tax_working
            l3 *= (
                1
                + self.params.equity_mean
                - self.params.fund_fee
                - self.params.pension_fee
            )
            l3 += ref_
            l3_bs += ref_
            c *= 1.02
        rp *= (
            1 + self.params.equity_mean - self.params.fund_fee - self.params.pension_fee
        )
        l3 *= (
            1 + self.params.equity_mean - self.params.fund_fee - self.params.pension_fee
        )
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
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        if current_year < self.params.glide_path_years:
            frac = 1.0 / self.params.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        needed = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed, p.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot


class ScenarioL3Broker(Scenario):
    """50% L3, 50% Broker => unify them at retirement so eq->bd decum."""

    def accumulate(self) -> Pot:
        pot = Pot()
        l3_eq = 0.0
        l3_eq_bs = 0.0
        br_eq, br_bd = 0.0, 0.0
        br_eq_bs, br_bd_bs = 0.0, 0.0
        c = self.params.annual_contribution
        proportion_l3 = 0.4

        for _ in range(self.params.years_accum):
            l3_eq *= (
                1
                + self.params.equity_mean
                - self.params.fund_fee
                - self.params.pension_fee
            )
            br_eq *= 1 + self.params.equity_mean - self.params.fund_fee
            br_bd *= 1 + self.params.bond_mean - self.params.fund_fee

            c_l3 = c * proportion_l3
            c_br = c * (1 - proportion_l3)
            l3_eq += c_l3
            l3_eq_bs += c_l3
            br_eq += c_br
            br_eq_bs += c_br
            c *= 1.02

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
        age_now = self.params.age_retire + current_year
        if age_now >= 70 and (age_now - 70) < self.params.glide_path_years:
            frac = 1.0 / self.params.glide_path_years
            shift_equity_to_bonds(pot, frac, self.params.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        # partial withdrawal
        withdrawn = withdraw(
            pot,
            max(0, needed_net - net_ann),
            self.params.cg_tax_normal,
        )
        return withdrawn.net_withdrawn + net_ann, pot


# --------------------------------------------------------
# 8. RUNNER
# --------------------------------------------------------
def simulate_montecarlo(scenario: Scenario):
    p = scenario.params
    init_pot = accumulate_initial_pots(p)
    scen_pot = scenario.accumulate()
    combined_pot = Pot(
        rurup=init_pot.rurup + scen_pot.rurup,
        l3_eq=init_pot.l3_eq + scen_pot.l3_eq,
        l3_eq_bs=init_pot.l3_eq_bs + scen_pot.l3_eq_bs,
        br_eq=init_pot.br_eq + scen_pot.br_eq,
        br_eq_bs=init_pot.br_eq_bs + scen_pot.br_eq_bs,
        br_bd=init_pot.br_bd + scen_pot.br_bd,
        br_bd_bs=init_pot.br_bd_bs + scen_pot.br_bd_bs,
    )
    lifetimes = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire
    results = []
    leftover_pots = []
    outcount = 0

    for i in range(p.num_sims):
        sim_pot = replace(combined_pot)  # copy
        total_spend = 0.0
        ran_out = False
        spend = p.desired_spend
        T = lifetimes[i]

        for year in range(p.years_transition):
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)
            scenario.transition_year(sim_pot, year, eq_r, bd_r)

        net_ann = scenario.params.public_pension
        if sim_pot.rurup > 0:
            gross_ann = sim_pot.rurup * p.ruerup_ann_rate
            net_ann += gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
            sim_pot.rurup = 0

        for t in range(T):
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)

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


# --------------------------------------------------------
# 9. PLOT
# --------------------------------------------------------
def plot_boxplot(data_list, labels):
    all_data = [d["all_spend"] for d in data_list]
    plt.figure(figsize=(7, 4))
    plt.boxplot(all_data, tick_labels=labels)
    plt.title("Boxplot of Discounted Spending (Sorted by leftover pot)")
    plt.ylabel("NPV of Spending")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------
# 10. MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    p = Params()

    # Create scenarios
    sA = ScenarioBroker(p)
    sB = ScenarioRurupBroker(p)
    sC = ScenarioL3(p)
    sD = ScenarioRurupL3(p)
    sE = ScenarioL3Broker(p)

    # simulate
    rA = simulate_montecarlo(sA)
    rB = simulate_montecarlo(sB)
    rC = simulate_montecarlo(sC)
    rD = simulate_montecarlo(sD)
    rE = simulate_montecarlo(sE)

    # Sort by p50pot
    combos = [
        ("Broker", rA),
        ("RurupBroker", rB),
        ("L3", rC),
        ("RurupL3", rD),
        ("L3Broker", rE),
    ]
    combos_sorted = sorted(combos, key=lambda x: x[1]["p50pot"], reverse=True)

    for name, res in combos_sorted:
        print(
            f"Scenario {name}: leftover p50={res['p50pot']:.0f}, run-out={res['prob_runout']*100:.1f}%, "
            f"p50 spend={res['p50']:,.0f}"
        )

    # box plot
    plot_boxplot([c[1] for c in combos_sorted], [c[0] for c in combos_sorted])
