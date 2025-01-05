import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from params import Params
from dataclasses import dataclass


@dataclass
class Pot:
    rurup: float = 0.0

    br_eq: float = 0.0
    br_eq_bs: float = 0.0
    br_bd: float = 0.0
    br_bd_bs: float = 0.0

    l3_eq: float = 0.0
    l3_eq_bs: float = 0.0

    # net annuity
    net_ann: float = 0.0

    def leftover(self):
        return self.br_eq + self.br_bd + self.l3_eq + self.rurup + self.net_ann


@dataclass
class Withdrawal:
    gross_withdrawn: float
    net_withdrawn: float


# --------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------
def sample_lifetime_from67(gender="M", size=10_000):
    ages = np.arange(67, 106)
    if gender == "M":
        pdf = np.array(
            [
                0.004,
                0.004,
                0.005,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.035,
                0.045,
                0.050,
                0.055,
                0.065,
                0.070,
                0.080,
                0.085,
                0.090,
                0.090,
                0.085,
                0.075,
                0.065,
                0.050,
                0.040,
                0.030,
                0.025,
                0.020,
                0.015,
                0.010,
                0.008,
                0.006,
                0.004,
                0.003,
                0.003,
                0.002,
                0.002,
                0.002,
                0.001,
                0.001,
            ]
        )
    else:
        pdf = np.array(
            [
                0.003,
                0.003,
                0.004,
                0.006,
                0.007,
                0.010,
                0.014,
                0.018,
                0.024,
                0.032,
                0.040,
                0.045,
                0.050,
                0.060,
                0.065,
                0.075,
                0.080,
                0.085,
                0.090,
                0.085,
                0.075,
                0.065,
                0.055,
                0.045,
                0.035,
                0.028,
                0.022,
                0.016,
                0.012,
                0.009,
                0.006,
                0.004,
                0.003,
                0.003,
                0.002,
                0.002,
                0.002,
                0.001,
                0.001,
            ]
        )
    pdf = pdf / pdf.sum()
    return np.random.choice(ages, p=pdf, size=size)


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
            pot.br_bd_bs = pot.br_eq_bs * (1 - frac2)
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
    def accumulate(self, p: Params) -> Pot:
        raise NotImplementedError

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        raise NotImplementedError


# --------------------------------------------------------
# 5. ACCUMULATE INITIAL POTS
# --------------------------------------------------------
def accumulate_initial_pots(p: Params) -> Pot:
    """Grow the user's initial pots from age_start..age_retire."""
    pot = Pot()
    for _ in range(p.years_accum):
        pot.rurup *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        pot.br_eq *= 1 + p.equity_mean - p.fund_fee
        pot.l3_eq *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
    pot.br_eq = p.initial_broker_bs
    pot.l3_eq_bs = p.initial_l3_bs
    return pot


def prepare_decum(init_pot: Pot, scenario_pot: Pot, params: Params) -> Pot:
    """
    Unify init_pot into pot_scenario so that at retirement
    we have exactly eq, eq_bs, bd, bd_bs (plus net_ann if any).
    This avoids leftover sub-pots that never decumulate.
    """
    final = Pot()

    # 1) Convert Rürup => net annuity
    final.rurup += scenario_pot.rurup
    final.rurup += init_pot.rurup
    if final.rurup > 0:
        gross_ann = final.rurup * params.ruerup_ann_rate
        net_ann_init = gross_ann * (1 - params.ruerup_tax - params.ruerup_dist_fee)
        final.net_ann += scenario_pot.net_ann + net_ann_init
        final.rurup = 0

    # 2) Convert L3 => lumpsum half CG => eq
    final.l3_eq += scenario_pot.l3_eq
    final.l3_eq += init_pot.l3_eq
    if final.l3_eq > 0:
        gains = max(0, final.l3_eq - final.l3_eq_bs)
        tax_ = gains * params.cg_tax_half
        net_l3 = max(0, final.l3_eq - tax_)
        final.br_eq += net_l3
        final.br_eq_bs += net_l3
        final.l3_eq = 0
        final.l3_eq_bs = 0

    final.net_ann += scenario_pot.net_ann
    final.br_eq += scenario_pot.br_eq
    final.br_eq_bs += scenario_pot.br_eq_bs
    final.br_bd += scenario_pot.br_bd
    final.br_bd_bs += scenario_pot.br_bd_bs

    return final


class ScenarioBroker(Scenario):
    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        eq_val = 0.0
        bs_val = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            eq_val *= 1 + p.equity_mean - p.fund_fee
            eq_val += ann_contr
            bs_val += ann_contr
            ann_contr *= 1.02

        eq_val *= 1 + p.equity_mean - p.fund_fee
        pot.br_eq = eq_val
        pot.br_eq_bs = bs_val
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            shift_equity_to_bonds(
                pot,
                frac,
                p.cg_tax_normal,
            )

        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.br_eq *= 1 + eq_r - p.fund_fee
        pot.br_bd *= 1 + bd_r - p.fund_fee

        withdrawn = withdraw(
            pot,
            needed_net,
            p.cg_tax_normal,
        )
        return withdrawn.net_withdrawn, pot


class ScenarioRurupBroker(Scenario):
    """Rürup + Broker => invests refund in broker pot."""

    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        rp = 0.0
        br = 0.0
        br_bs = 0.0
        ann_c = p.annual_contribution
        for _ in range(p.years_accum):
            rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            rp += ann_c
            # broker invests refund
            ref_ = ann_c * p.tax_working
            br *= 1 + p.equity_mean - p.fund_fee
            br += ref_
            br_bs += ref_

            ann_c *= 1.02

        rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        br *= 1 + p.equity_mean - p.fund_fee

        # store rp in eq, unify at retirement to net_ann
        pot.rurup = rp
        pot.br_eq = br
        pot.br_eq_bs = br_bs
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        net_ann = pot.net_ann
        eq = pot.br_eq
        bd = pot.br_bd
        eq_bs = pot.br_eq_bs
        bd_bs = pot.br_bd_bs

        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            shift_equity_to_bonds(pot, frac, p.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        needed_broker = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed_broker, p.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot


class ScenarioL3(Scenario):
    """L3 lumpsum half CG => eq->bd decum."""

    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        val, bs = 0.0, 0.0
        c = p.annual_contribution
        for _ in range(p.years_accum):
            val *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            val += c
            bs += c
            c *= 1.02
        val *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        pot.l3_eq = val
        pot.l3_eq_bs = bs
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
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
            needed_net,
            p.cg_tax_normal,
        )
        return withdrawn.net_withdrawn, pot


class ScenarioRurupL3(Scenario):
    """Rürup + L3 lumpsum => eq->bond decum."""

    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        rp, l3, l3_bs = 0.0, 0.0, 0.0
        c = p.annual_contribution
        for _ in range(p.years_accum):
            rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            rp += c
            # L3 invests the refund
            ref_ = c * p.tax_working
            l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3 += ref_
            l3_bs += ref_
            c *= 1.02
        rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        pot.rurup = rp
        pot.l3_eq = l3
        pot.l3_eq_bs = l3_bs
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        net_ann = pot.net_ann
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            shift_equity_to_bonds(pot, frac, p.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        needed = max(0, needed_net - net_ann)
        withdrawn = withdraw(pot, needed, p.cg_tax_normal)
        total_net = net_ann + withdrawn.net_withdrawn
        return total_net, pot


class ScenarioL3Broker(Scenario):
    """50% L3, 50% Broker => unify them at retirement so eq->bd decum."""

    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        l3_eq, l3_bd = 0.0, 0.0
        l3_eq_bs, l3_bd_bs = 0.0, 0.0
        br_eq, br_bd = 0.0, 0.0
        br_eq_bs, br_bd_bs = 0.0, 0.0
        c = p.annual_contribution
        half = 0.5

        for _ in range(p.years_accum):
            l3_eq *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3_bd *= 1 + p.bond_mean - p.fund_fee - p.pension_fee
            br_eq *= 1 + p.equity_mean - p.fund_fee
            br_bd *= 1 + p.bond_mean - p.fund_fee

            c_l3 = c * half
            c_br = c * (1 - half)
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
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        # SHIFT eq->bd at ages 72..(72+p.glide_path_years)
        age_now = p.age_retire + current_year
        if age_now >= 72 and (age_now - 72) < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            shift_equity_to_bonds(pot, frac, p.cg_tax_normal)

        pot.br_eq *= 1 + rand_returns["eq"]
        pot.br_bd *= 1 + rand_returns["bd"]

        # partial withdrawal
        withdrawn = withdraw(
            pot,
            needed_net,
            p.cg_tax_normal,
        )
        return withdrawn.net_withdrawn, pot


# --------------------------------------------------------
# 8. RUNNER
# --------------------------------------------------------
def simulate_montecarlo(scenario: Scenario, p: Params):
    init_pot = accumulate_initial_pots(p)
    scen_pot = scenario.accumulate(p)
    final_pot = prepare_decum(init_pot, scen_pot, p)
    lifetimes = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire
    results = []
    leftover_pots = []
    outcount = 0

    for i in range(p.num_sims):
        sim_pot = replace(final_pot)  # copy
        total_spend = 0.0
        ran_out = False
        spend = p.desired_spend
        T = lifetimes[i]

        for t in range(T):
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)

            net_wd, sim_pot = scenario.decumulate_year(
                sim_pot, p, t, spend, rand_returns={"eq": eq_r, "bd": bd_r}
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
    sA = ScenarioBroker()
    sB = ScenarioRurupBroker()
    sC = ScenarioL3()
    sD = ScenarioRurupL3()
    sE = ScenarioL3Broker()

    # simulate
    rA = simulate_montecarlo(sA, p)
    rB = simulate_montecarlo(sB, p)
    rC = simulate_montecarlo(sC, p)
    rD = simulate_montecarlo(sD, p)
    rE = simulate_montecarlo(sE, p)

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
