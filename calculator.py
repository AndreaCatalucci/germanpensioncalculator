import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from params import Params


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


def shift_equity_to_bonds(eq, eq_bs, bd, bd_bs, fraction, cgt):
    moved = eq * fraction
    bs_moved = eq_bs * fraction
    eq_after = eq - moved
    eq_bs_after = eq_bs - bs_moved
    gains = moved - bs_moved
    tax_amount = gains * cgt
    moved_net = moved - tax_amount
    bd_after = bd + moved_net
    bd_bs_after = bd_bs + moved_net
    return eq_after, eq_bs_after, bd_after, bd_bs_after


def solve_gross_for_net(eq, bd, eq_bs, bd_bs, net_needed, cg_tax):
    total_pot = eq + bd
    if total_pot <= 0:
        return (0, 0, 0, 0, 0, 0)

    net_withdrawn = 0
    gross_withdrawn = 0
    bd_after = bd
    eq_after = eq
    bd_bs_after = bd_bs
    eq_bs_after = eq_bs
    still_needed = net_needed

    # BOND FIRST
    if bd > 0:
        net_bd = bd - (bd - bd_bs) * cg_tax
        portion = min(still_needed, net_bd)
        if portion > 0:
            frac = portion / net_bd
            gross_bd_withdrawal = frac * bd
            bd_bs_after = bd_bs * (1 - frac)
            bd_after = bd - gross_bd_withdrawal
            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # EQUITY NEXT
    if still_needed > 0 and eq_after > 0:
        net_eq = eq_after - (eq_after - eq_bs_after) * cg_tax
        portion2 = min(still_needed, net_eq)
        if portion2 > 0:
            frac2 = portion2 / net_eq
            gross_eq_withdrawal = frac2 * eq_after
            eq_bs_after = eq_bs_after * (1 - frac2)
            eq_after = eq_after - gross_eq_withdrawal
            net_withdrawn += portion2
            gross_withdrawn += gross_eq_withdrawal
            still_needed -= portion2

    return (
        gross_withdrawn,
        net_withdrawn,
        eq_after,
        bd_after,
        eq_bs_after,
        bd_bs_after,
    )


# --------------------------------------------------------
# 3. POT DATACLASS
# --------------------------------------------------------
from dataclasses import dataclass


@dataclass
class Pot:
    eq: float = 0.0
    eq_bs: float = 0.0
    bd: float = 0.0
    bd_bs: float = 0.0

    # Possibly for Rürup or L3, but we'll unify them at retirement:
    br_eq: float = 0.0
    br_eq_bs: float = 0.0
    br_bd: float = 0.0
    br_bd_bs: float = 0.0

    l3_eq: float = 0.0
    l3_eq_bs: float = 0.0
    l3_bd: float = 0.0
    l3_bd_bs: float = 0.0

    # net annuity
    net_ann: float = 0.0

    def leftover(self):
        """Sum only the principal in eq/bd + br_eq/br_bd + l3_eq/l3_bd."""
        return self.eq + self.bd + self.br_eq + self.br_bd + self.l3_eq + self.l3_bd


# --------------------------------------------------------
# 4. BASE SCENARIO
# --------------------------------------------------------
class Scenario:
    def accumulate(self, p: Params) -> Pot:
        raise NotImplementedError

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        raise NotImplementedError


# --------------------------------------------------------
# 5. ACCUMULATE INITIAL POTS
# --------------------------------------------------------
def accumulate_initial_pots(p: Params) -> Pot:
    """Grow the user's initial pots from age_start..age_retire."""
    pot = Pot()

    # We'll store the final amounts in pot's 'br_eq' for Broker
    # 'eq' for Rürup principal, 'l3_eq' for L3, then unify later
    # or do the approach below:
    # For clarity, let's store them as we see them:
    r = p.initial_rurup
    br = p.initial_broker
    br_bs = p.initial_broker_bs
    l3 = p.initial_l3
    l3_bs = p.initial_l3_bs

    for _ in range(p.years_accum):
        r *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        br *= 1 + p.equity_mean - p.fund_fee
        l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee

    pot.eq = r  # We'll treat eq as the final Rürup principal
    pot.br_eq = br
    pot.br_eq_bs = br_bs
    pot.l3_eq = l3
    pot.l3_eq_bs = l3_bs

    return pot


# --------------------------------------------------------
# 6. MERGE PREEXISTING POTS => SCENARIO POT
# --------------------------------------------------------
def add_initial_pots(pot_scenario: Pot, p: Params, init_pot: Pot) -> Pot:
    """
    Unify init_pot into pot_scenario so that at retirement
    we have exactly eq, eq_bs, bd, bd_bs (plus net_ann if any).
    This avoids leftover sub-pots that never decumulate.
    """
    final = Pot()

    # 1) Convert Rürup => net annuity
    # init_pot.eq is the final rürup principal
    if init_pot.eq > 0:
        # we treat that as rürup principal
        gross_ann = init_pot.eq * p.ruerup_ann_rate
        net_ann_init = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
        final.net_ann = pot_scenario.net_ann + net_ann_init

    # 2) Convert L3 => lumpsum half CG => eq
    if init_pot.l3_eq > 0:
        gains = max(0, init_pot.l3_eq - init_pot.l3_eq_bs)
        tax_ = gains * p.cg_tax_half
        net_l3 = max(0, init_pot.l3_eq - tax_)
        # add to eq
        final.eq = pot_scenario.eq + net_l3
        final.eq_bs = pot_scenario.eq_bs + net_l3
    else:
        final.eq = pot_scenario.eq
        final.eq_bs = pot_scenario.eq_bs

    # 3) Broker => add to eq
    final.eq += init_pot.br_eq
    final.eq_bs += init_pot.br_eq_bs

    # 4) unify scenario pot's net_ann, eq, bd, etc.
    # If scenario pot has eq/bd leftover
    # plus scenario pot's net_ann
    final.net_ann += pot_scenario.net_ann
    final.eq += pot_scenario.eq
    final.eq_bs += pot_scenario.eq_bs
    final.bd += pot_scenario.bd
    final.bd_bs += pot_scenario.bd_bs

    # If scenario pot has br pot, lumpsum that too
    final.eq += pot_scenario.br_eq
    final.eq_bs += pot_scenario.br_eq_bs
    final.bd += pot_scenario.br_bd
    final.bd_bs += pot_scenario.br_bd_bs

    # If scenario pot has L3 eq, lumpsum half CG it?
    if pot_scenario.l3_eq > 0:
        gains2 = max(0, pot_scenario.l3_eq - pot_scenario.l3_eq_bs)
        tax2 = gains2 * p.cg_tax_half
        net2 = max(0, pot_scenario.l3_eq - tax2)
        final.eq += net2
        final.eq_bs += net2

    final.l3_eq = 0
    final.l3_eq_bs = 0
    final.l3_bd = 0
    final.l3_bd_bs = 0
    final.br_eq = 0
    final.br_eq_bs = 0
    final.br_bd = 0
    final.br_bd_bs = 0

    return final


# --------------------------------------------------------
# 7. SCENARIOS (SAME CODE, BUT PREPARE_DECUM UNIFIES AT RETIREMENT)
# --------------------------------------------------------
class ScenarioA(Scenario):
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
        pot.eq = eq_val
        pot.eq_bs = bs_val
        return pot

    # no special lumpsum => do nothing
    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            pot.eq, pot.eq_bs, pot.bd, pot.bd_bs = shift_equity_to_bonds(
                pot.eq, pot.eq_bs, pot.bd, pot.bd_bs, frac, p.cg_tax_normal
            )

        eq_r = rand_returns["eq"]
        bd_r = rand_returns["bd"]
        pot.eq *= 1 + eq_r
        pot.bd *= 1 + bd_r

        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            pot.eq, pot.bd, pot.eq_bs, pot.bd_bs, needed_net, p.cg_tax_normal
        )

        pot.eq, pot.bd = eq_after, bd_after
        pot.eq_bs, pot.bd_bs = eq_bs_after, bd_bs_after
        return net_, pot


class ScenarioB(Scenario):
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
        pot.eq = rp
        pot.br_eq = br
        pot.br_eq_bs = br_bs
        return pot

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        # convert eq => net ann (Rürup)
        rp = pot.eq
        gross_ann = rp * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
        pot.net_ann = net_ann
        # eq is now 0
        pot.eq = 0
        pot.eq_bs = 0
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        net_ann = pot.net_ann
        eq = pot.br_eq
        bd = pot.br_bd
        eq_bs = pot.br_eq_bs
        bd_bs = pot.br_bd_bs

        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, frac, p.cg_tax_normal
            )

        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        needed_broker = max(0, needed_net - net_ann)
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_broker, p.cg_tax_normal
        )
        total_net = net_ann + net_wd

        pot.br_eq = eq_after
        pot.br_bd = bd_after
        pot.br_eq_bs = eq_bs_after
        pot.br_bd_bs = bd_bs_after
        return total_net, pot


class ScenarioC(Scenario):
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

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        # lumpsum half CG => eq
        lumpsum = pot.l3_eq
        bs = pot.l3_eq_bs
        gains = max(0, lumpsum - bs)
        tax_ = gains * p.cg_tax_half
        dist_fee = lumpsum * p.ruerup_dist_fee
        net_ = max(0, lumpsum - tax_ - dist_fee)
        pot.eq = net_
        pot.eq_bs = net_
        pot.l3_eq = 0
        pot.l3_eq_bs = 0
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            pot.eq, pot.eq_bs, pot.bd, pot.bd_bs = shift_equity_to_bonds(
                pot.eq, pot.eq_bs, pot.bd, pot.bd_bs, frac, p.cg_tax_normal
            )

        pot.eq *= 1 + rand_returns["eq"]
        pot.bd *= 1 + rand_returns["bd"]

        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            pot.eq, pot.bd, pot.eq_bs, pot.bd_bs, needed_net, p.cg_tax_normal
        )
        pot.eq, pot.bd = eq_after, bd_after
        pot.eq_bs, pot.bd_bs = eq_bs_after, bd_bs_after
        return net_, pot


class ScenarioD(Scenario):
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
        pot.eq = rp  # treat eq as Rürup
        pot.l3_eq = l3
        pot.l3_eq_bs = l3_bs
        return pot

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        rp = pot.eq
        gross_ann = rp * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
        pot.net_ann = net_ann
        pot.eq = 0
        pot.eq_bs = 0
        # lumpsum L3
        l3 = pot.l3_eq
        bs = pot.l3_eq_bs
        gains = max(0, l3 - bs)
        tax_ = gains * p.cg_tax_half
        net_l3 = max(0, l3 - tax_)
        pot.eq = net_l3
        pot.eq_bs = net_l3
        pot.l3_eq = 0
        pot.l3_eq_bs = 0
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        net_ann = pot.net_ann
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            pot.eq, pot.eq_bs, pot.bd, pot.bd_bs = shift_equity_to_bonds(
                pot.eq, pot.eq_bs, pot.bd, pot.bd_bs, frac, p.cg_tax_normal
            )

        pot.eq *= 1 + rand_returns["eq"]
        pot.bd *= 1 + rand_returns["bd"]

        needed = max(0, needed_net - net_ann)
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            pot.eq, pot.bd, pot.eq_bs, pot.bd_bs, needed, p.cg_tax_normal
        )
        total_net = net_ann + net_wd

        pot.eq, pot.bd = eq_after, bd_after
        pot.eq_bs, pot.bd_bs = eq_bs_after, bd_bs_after
        return total_net, pot


class ScenarioE(Scenario):
    """L3 => annuity => leftover=0 => run-out if leftover check is 0?"""

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

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        # entire L3 => annuity taxed at 17%?
        val = pot.l3_eq
        gross_ann = val * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax * 0.17 - p.ruerup_dist_fee)
        pot.net_ann = net_ann
        pot.l3_eq = 0
        pot.l3_eq_bs = 0
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        # leftover pot=0 => immediate run-out if leftover<0
        # The code sees leftover=0 => run-out
        return pot.net_ann, pot


class ScenarioF(Scenario):
    """50% L3, 50% Broker => unify them at retirement so eq->bd decum."""

    def accumulate(self, p: Params) -> Pot:
        pot = Pot()
        l3_eq, l3_bd = 0.0, 0.0
        l3_eq_bs, l3_bd_bs = 0.0, 0.0
        br_eq, br_bd = 0.0, 0.0
        br_eq_bs, br_bd_bs = 0.0, 0.0
        c = p.annual_contribution
        half = 0.5

        for yr in range(p.years_accum):
            age_now = p.age_start + yr
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

            if 62 <= age_now < 67:
                frac = 1.0 / 5.0
                bd_before = l3_bd
                l3_eq, l3_eq_bs, l3_bd, l3_bd_bs = shift_equity_to_bonds(
                    l3_eq, l3_eq_bs, l3_bd, l3_bd_bs, frac, p.cg_tax_half
                )
                delta = l3_bd - bd_before
                l3_bd -= delta * p.ruerup_dist_fee

        pot.l3_eq, pot.l3_bd = l3_eq, l3_bd
        pot.l3_eq_bs, pot.l3_bd_bs = l3_eq_bs, l3_bd_bs
        pot.br_eq, pot.br_bd = br_eq, br_bd
        pot.br_eq_bs, pot.br_bd_bs = br_eq_bs, br_bd_bs
        return pot

    def prepare_decum(self, pot: Pot, p: Params) -> Pot:
        # unify them so eq,bd are used in decum
        # lumpsum L3 => half CG
        l3_gains = max(0, pot.l3_eq - pot.l3_eq_bs)
        tax_ = l3_gains * p.cg_tax_half
        net_l3 = max(0, pot.l3_eq - tax_)
        pot.eq += net_l3
        pot.eq_bs += net_l3

        # unify broker eq-> eq
        pot.eq += pot.br_eq
        pot.eq_bs += pot.br_eq_bs
        pot.bd += pot.br_bd
        pot.bd_bs += pot.br_bd_bs

        # zero out sub-pots
        pot.br_eq = 0
        pot.br_eq_bs = 0
        pot.br_bd = 0
        pot.br_bd_bs = 0
        pot.l3_eq = 0
        pot.l3_eq_bs = 0
        pot.l3_bd = 0
        pot.l3_bd_bs = 0
        return pot

    def decumulate_year(
        self,
        pot: Pot,
        p: Params,
        current_year: int,
        needed_net: float,
        rand_returns: dict,
    ) -> (float, Pot):
        # SHIFT eq->bd at ages 72..(72+p.glide_path_years)
        age_now = p.age_retire + current_year
        if age_now >= 72 and (age_now - 72) < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            pot.eq, pot.eq_bs, pot.bd, pot.bd_bs = shift_equity_to_bonds(
                pot.eq, pot.eq_bs, pot.bd, pot.bd_bs, frac, p.cg_tax_normal
            )

        pot.eq *= 1 + rand_returns["eq"]
        pot.bd *= 1 + rand_returns["bd"]

        # partial withdrawal
        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            pot.eq, pot.bd, pot.eq_bs, pot.bd_bs, needed_net, p.cg_tax_normal
        )
        pot.eq, pot.bd = eq_after, bd_after
        pot.eq_bs, pot.bd_bs = eq_bs_after, bd_bs_after
        return net_, pot


# --------------------------------------------------------
# 8. RUNNER
# --------------------------------------------------------
def simulate_montecarlo(scenario: Scenario, p: Params):
    init_pot = accumulate_initial_pots(p)
    scen_pot = scenario.accumulate(p)
    scen_pot = scenario.prepare_decum(scen_pot, p)
    final_pot = add_initial_pots(scen_pot, p, init_pot)

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

            infl = np.random.normal(p.inflation_mean, p.inflation_std)
            spend *= 1 + infl

            if sim_pot.leftover() <= 0:
                ran_out = True
                break

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
    plt.boxplot(all_data, labels=labels)
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
    sA = ScenarioA()
    sB = ScenarioB()
    sC = ScenarioC()
    sD = ScenarioD()
    sE = ScenarioE()
    sF = ScenarioF()

    # simulate
    rA = simulate_montecarlo(sA, p)
    rB = simulate_montecarlo(sB, p)
    rC = simulate_montecarlo(sC, p)
    rD = simulate_montecarlo(sD, p)
    rE = simulate_montecarlo(sE, p)
    rF = simulate_montecarlo(sF, p)

    # Sort by p50pot
    combos = [
        ("A", rA),
        ("B", rB),
        ("C", rC),
        ("D", rD),
        ("E", rE),
        ("F", rF),
    ]
    combos_sorted = sorted(combos, key=lambda x: x[1]["p50pot"], reverse=True)

    for name, res in combos_sorted:
        print(
            f"Scenario {name}: leftover p50={res['p50pot']:.0f}, run-out={res['prob_runout']*100:.1f}%, "
            f"p50 spend={res['p50']:,.0f}"
        )

    # box plot
    plot_boxplot([c[1] for c in combos_sorted], [c[0] for c in combos_sorted])
