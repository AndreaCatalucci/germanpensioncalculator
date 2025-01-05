import numpy as np
import matplotlib.pyplot as plt
from params import Params


# --------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------
def sample_lifetime_from67(gender="M", size=10_000):
    """
    Returns an array of 'size' random ages of death, at or after age 67,
    with a distribution that is more optimistic for 2060 Germany.
    """
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
    else:  # "F"
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


def shift_equity_to_bonds(eq, eq_bs, bd, bd_bs, fraction, cgt):
    """Shift 'fraction' of eq -> bd, paying cgt on the gains portion."""
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


def present_value(cf, year, discount_rate):
    """Discount a cash flow 'cf' that occurs 'year' years after retirement=0."""
    return cf / ((1 + discount_rate) ** year)


def solve_gross_for_net(eq, bd, eq_basis, bd_basis, net_needed, cg_tax):
    """
    Withdraw 'net_needed' from eq+bd, paying cg_tax on gains.
    BONDS first, then EQUITY.
    """
    total_pot = eq + bd
    if total_pot <= 0:
        return (0, 0, 0, 0, 0, 0)

    net_withdrawn = 0
    gross_withdrawn = 0

    bd_after = bd
    eq_after = eq
    bd_bs_after = bd_basis
    eq_bs_after = eq_basis

    still_needed = net_needed

    # 1) BOND
    if bd > 0:
        net_bd = bd - (bd - bd_basis) * cg_tax
        portion = min(still_needed, net_bd)
        if portion > 0:
            frac = portion / net_bd
            gross_bd_withdrawal = frac * bd
            bd_bs_after = bd_basis * (1 - frac)
            bd_after = bd - gross_bd_withdrawal

            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # 2) EQUITY
    if (still_needed > 0) and (eq_after > 0):
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
# 3. BASE SCENARIO CLASS
# --------------------------------------------------------
class Scenario:
    """
    Base scenario, dealing only with *new contributions*
    (no references to p.initial_* here).
    """

    def accumulate(self, p: Params):
        raise NotImplementedError

    def prepare_decum(self, pot_dict, p: Params):
        return pot_dict

    def decumulate_year(
        self, pot_dict, p: Params, current_year, needed_net, rand_returns
    ):
        raise NotImplementedError


# --------------------------------------------------------
# 4. ACCUMULATE INITIAL POTS
# --------------------------------------------------------
def accumulate_initial_pots(p: Params):
    """
    Grows initial Rürup, Broker, L3 from age_start..age_retire
    with eq returns (minus fees).
    - Rürup => eq mean - fund_fee - pension_fee
    - Broker => eq mean - fund_fee
    - L3 => eq mean - fund_fee - pension_fee
    """
    r = p.initial_rurup
    br = p.initial_broker
    br_bs = p.initial_broker_bs
    l3 = p.initial_l3
    l3_bs = p.initial_l3_bs

    for _ in range(p.years_accum):
        # Rürup
        r *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        # Broker
        br *= 1 + p.equity_mean - p.fund_fee
        # L3
        l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee

    return {
        "rurup": r,
        "broker": br,
        "broker_bs": br_bs,  # basis doesn't grow
        "l3": l3,
        "l3_bs": l3_bs,  # basis doesn't grow
    }


# --------------------------------------------------------
# 5. MERGE PREEXISTING POTS
# --------------------------------------------------------
def add_initial_pots(pot_dict: dict, p: Params, init_dict: dict) -> dict:
    """
    Merges preexisting final pots into scenario pot at retirement:
      - Rürup => net annuity
      - Broker => eq/eq_bs
      - L3 => lumpsum half CG => eq/eq_bs
    """
    # Rürup => annuity
    if init_dict["rurup"] > 0:
        gross_ann = init_dict["rurup"] * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)
        pot_dict["net_ann"] = pot_dict.get("net_ann", 0.0) + net_ann

    # Broker => eq pot
    eq_curr = pot_dict.get("eq", 0.0)
    eq_bs_curr = pot_dict.get("eq_bs", 0.0)

    eq_curr += init_dict["broker"]
    eq_bs_curr += init_dict["broker_bs"]

    pot_dict["eq"] = eq_curr
    pot_dict["eq_bs"] = eq_bs_curr

    if "bd" not in pot_dict:
        pot_dict["bd"] = 0.0
    if "bd_bs" not in pot_dict:
        pot_dict["bd_bs"] = 0.0

    # L3 => lumpsum half CG
    if init_dict["l3"] > 0:
        gains = max(0, init_dict["l3"] - init_dict["l3_bs"])
        tax_ = gains * p.cg_tax_half
        net_l3 = max(0, init_dict["l3"] - tax_)

        pot_dict["eq"] += net_l3
        pot_dict["eq_bs"] += net_l3

    return pot_dict


# --------------------------------------------------------
# 6. SCENARIOS A-F
# --------------------------------------------------------


# SCENARIO A
class ScenarioA(Scenario):
    """Broker Only => eq->bond decum"""

    def accumulate(self, p: Params):
        pot = 0.0
        basis = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee
            pot += ann_contr
            basis += ann_contr
            ann_contr *= 1.02  # optional inflation of contributions

        # final year returns
        pot *= 1 + p.equity_mean - p.fund_fee

        return {"eq": pot, "eq_bs": basis, "bd": 0.0, "bd_bs": 0.0}

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        # SHIFT eq->bond
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, frac, p.cg_tax_normal
            )

        # random returns
        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        # partial withdrawal
        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_net, p.cg_tax_normal
        )

        pot_dict["eq"] = eq_after
        pot_dict["bd"] = bd_after
        pot_dict["eq_bs"] = eq_bs_after
        pot_dict["bd_bs"] = bd_bs_after

        return net_, pot_dict


# SCENARIO B
class ScenarioB(Scenario):
    """
    Rürup + Broker
    - Rürup pot => net annuity
    - Broker eq->bond decum
    """

    def accumulate(self, p: Params):
        rpot = 0.0
        br = 0.0
        br_bs = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            rpot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            rpot += ann_contr

            # broker invests "refund"
            refund = ann_contr * p.tax_working
            br *= 1 + p.equity_mean - p.fund_fee
            br += refund
            br_bs += refund

            ann_contr *= 1.02

        rpot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        br *= 1 + p.equity_mean - p.fund_fee

        return {
            "rpot": rpot,
            "br_eq": br,
            "br_eq_bs": br_bs,
            "br_bd": 0.0,
            "br_bd_bs": 0.0,
        }

    def prepare_decum(self, pot_dict, p: Params):
        # convert rpot => net annuity
        rp = pot_dict["rpot"]
        gross_ann = rp * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)

        pot_dict["net_ann"] = pot_dict.get("net_ann", 0.0) + net_ann
        pot_dict.pop("rpot", None)
        return pot_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        net_ann = pot_dict.get("net_ann", 0.0)
        eq = pot_dict["br_eq"]
        bd = pot_dict["br_bd"]
        eq_bs = pot_dict["br_eq_bs"]
        bd_bs = pot_dict["br_bd_bs"]

        # SHIFT eq->bd
        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, frac, p.cg_tax_normal
            )

        # returns
        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        needed_broker = max(0, needed_net - net_ann)
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_broker, p.cg_tax_normal
        )

        total_net = net_ann + net_wd

        pot_dict["br_eq"] = eq_after
        pot_dict["br_bd"] = bd_after
        pot_dict["br_eq_bs"] = eq_bs_after
        pot_dict["br_bd_bs"] = bd_bs_after

        return total_net, pot_dict


# SCENARIO C
class ScenarioC(Scenario):
    """
    Level3 => lumpsum half CG => eq->bond decum
    """

    def accumulate(self, p: Params):
        pot = 0.0
        basis = 0.0
        ann_contr = p.annual_contribution
        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            pot += ann_contr
            basis += ann_contr
            ann_contr *= 1.02

        pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        return {"l3_pot": pot, "l3_bs": basis}

    def prepare_decum(self, pot_dict, p: Params):
        pot = pot_dict["l3_pot"]
        bs = pot_dict["l3_bs"]
        gains = max(0, pot - bs)
        tax_ = gains * p.cg_tax_half
        dist_fee_amt = pot * p.ruerup_dist_fee  # optional

        net_lump = max(0, pot - tax_ - dist_fee_amt)

        return {
            "eq": net_lump,
            "eq_bs": net_lump,
            "bd": 0.0,
            "bd_bs": 0.0,
        }

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, frac, p.cg_tax_normal
            )

        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_net, p.cg_tax_normal
        )

        pot_dict["eq"] = eq_after
        pot_dict["bd"] = bd_after
        pot_dict["eq_bs"] = eq_bs_after
        pot_dict["bd_bs"] = bd_bs_after

        return net_, pot_dict


# SCENARIO D
class ScenarioD(Scenario):
    """
    Rürup + L3 lumpsum => eq->bond decum
    """

    def accumulate(self, p: Params):
        rp = 0.0
        l3 = 0.0
        l3_bs = 0.0
        ann_contr = p.annual_contribution
        for _ in range(p.years_accum):
            # Rürup
            rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            rp += ann_contr

            # L3 invests 'refund'
            refund = ann_contr * p.tax_working
            l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3 += refund
            l3_bs += refund

            ann_contr *= 1.02

        rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee

        return {"rp": rp, "l3": l3, "l3_bs": l3_bs}

    def prepare_decum(self, pot_dict, p: Params):
        rp = pot_dict["rp"]
        gross_ann = rp * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)

        l3 = pot_dict["l3"]
        bs = pot_dict["l3_bs"]
        gains = max(0, l3 - bs)
        tax_ = gains * p.cg_tax_half
        net_l3 = max(0, l3 - tax_)

        return {
            "net_ann": net_ann,
            "eq": net_l3,
            "eq_bs": net_l3,
            "bd": 0.0,
            "bd_bs": 0.0,
        }

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        net_ann = pot_dict["net_ann"]
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        if current_year < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, frac, p.cg_tax_normal
            )

        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        needed = max(0, needed_net - net_ann)
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed, p.cg_tax_normal
        )

        total_net = net_ann + net_wd

        pot_dict["eq"] = eq_after
        pot_dict["bd"] = bd_after
        pot_dict["eq_bs"] = eq_bs_after
        pot_dict["bd_bs"] = bd_bs_after

        return total_net, pot_dict


# SCENARIO E
class ScenarioE(Scenario):
    """
    L3 as an annuity => taxed at 17% (example from snippet)
    """

    def accumulate(self, p: Params):
        pot = 0.0
        bs = 0.0
        ann_contr = p.annual_contribution
        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            pot += ann_contr
            bs += ann_contr
            ann_contr *= 1.02

        pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        return {"l3_pot": pot, "l3_bs": bs}

    def prepare_decum(self, pot_dict, p: Params):
        pot = pot_dict["l3_pot"]
        gross_ann = pot * p.ruerup_ann_rate
        # taxed at 17% => p.ruerup_tax * 0.17
        net_ann = gross_ann * (1 - p.ruerup_tax * 0.17 - p.ruerup_dist_fee)
        return {"net_ann": net_ann}

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        net_ann = pot_dict["net_ann"]
        # If needed_net > net_ann => partial run out
        return net_ann, pot_dict


# SCENARIO F
class ScenarioF(Scenario):
    """
    50% L3, 50% Broker
    L3 eq->bd from age 62..66,
    Broker eq->bd from age 72..(72+p.glide_path_years)
    """

    def accumulate(self, p: Params):
        l3_eq = 0.0
        l3_bd = 0.0
        l3_eq_bs = 0.0
        l3_bd_bs = 0.0

        br_eq = 0.0
        br_bd = 0.0
        br_eq_bs = 0.0
        br_bd_bs = 0.0

        ann_contr = p.annual_contribution
        half = 0.5

        for yr in range(p.years_accum):
            age_now = p.age_start + yr

            # returns
            l3_eq *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3_bd *= 1 + p.bond_mean - p.fund_fee - p.pension_fee

            br_eq *= 1 + p.equity_mean - p.fund_fee
            br_bd *= 1 + p.bond_mean - p.fund_fee

            # new contributions
            l3_contr = ann_contr * half
            br_contr = ann_contr * (1 - half)

            l3_eq += l3_contr
            l3_eq_bs += l3_contr
            br_eq += br_contr
            br_eq_bs += br_contr

            ann_contr *= 1.02

            # shift L3 eq->bd at age 62..66
            if 62 <= age_now < 67:
                frac = 1.0 / 5.0
                bd_before = l3_bd
                l3_eq, l3_eq_bs, l3_bd, l3_bd_bs = shift_equity_to_bonds(
                    l3_eq, l3_eq_bs, l3_bd, l3_bd_bs, frac, p.cg_tax_half
                )
                delta = l3_bd - bd_before
                # distribution fee
                l3_bd -= delta * p.ruerup_dist_fee

        return {
            "l3_eq": l3_eq,
            "l3_bd": l3_bd,
            "l3_eq_bs": l3_eq_bs,
            "l3_bd_bs": l3_bd_bs,
            "br_eq": br_eq,
            "br_bd": br_bd,
            "br_eq_bs": br_eq_bs,
            "br_bd_bs": br_bd_bs,
        }

    def prepare_decum(self, pot_dict, p: Params):
        # no lumpsum or annuity creation at 67
        return pot_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        eq_l3 = pot_dict["l3_eq"]
        bd_l3 = pot_dict["l3_bd"]
        eq_bs_l3 = pot_dict["l3_eq_bs"]
        bd_bs_l3 = pot_dict["l3_bd_bs"]

        eq_br = pot_dict["br_eq"]
        bd_br = pot_dict["br_bd"]
        eq_bs_br = pot_dict["br_eq_bs"]
        bd_bs_br = pot_dict["br_bd_bs"]

        # returns
        eq_l3 *= 1 + np.random.normal(p.equity_mean, p.equity_std)
        bd_l3 *= 1 + np.random.normal(p.bond_mean, p.bond_std)

        eq_br *= 1 + rand_returns["eq"]
        bd_br *= 1 + rand_returns["bd"]

        # shift broker eq->bd at age 72..(72 + p.glide_path_years)
        retired_age = p.age_retire + current_year
        if retired_age >= 72 and (retired_age - 72) < p.glide_path_years:
            frac = 1.0 / p.glide_path_years
            eq_br, eq_bs_br, bd_br, bd_bs_br = shift_equity_to_bonds(
                eq_br, eq_bs_br, bd_br, bd_bs_br, frac, p.cg_tax_normal
            )

        # combine for withdrawal
        eq_total = eq_l3 + eq_br
        bd_total = bd_l3 + bd_br
        eq_bs_total = eq_bs_l3 + eq_bs_br
        bd_bs_total = bd_bs_l3 + bd_bs_br

        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq_total, bd_total, eq_bs_total, bd_bs_total, needed_net, p.cg_tax_normal
        )

        # re-split
        eq_ratio_l3 = eq_l3 / eq_total if eq_total > 0 else 0
        eq_ratio_br = eq_br / eq_total if eq_total > 0 else 0
        bd_ratio_l3 = bd_l3 / bd_total if bd_total > 0 else 0
        bd_ratio_br = bd_br / bd_total if bd_total > 0 else 0

        pot_dict["l3_eq"] = eq_after * eq_ratio_l3
        pot_dict["l3_bd"] = bd_after * bd_ratio_l3
        pot_dict["l3_eq_bs"] = eq_bs_after * eq_ratio_l3
        pot_dict["l3_bd_bs"] = bd_bs_after * bd_ratio_l3

        pot_dict["br_eq"] = eq_after * eq_ratio_br
        pot_dict["br_bd"] = bd_after * bd_ratio_br
        pot_dict["br_eq_bs"] = eq_bs_after * eq_ratio_br
        pot_dict["br_bd_bs"] = bd_bs_after * bd_ratio_br

        return net_wd, pot_dict


# --------------------------------------------------------
# 7. MONTE CARLO RUNNER
# --------------------------------------------------------
def simulate_montecarlo(scenario: Scenario, p: Params):
    """
    Steps:
      1) accumulate_initial_pots => init_dict
      2) scenario.accumulate => scenario_dict
      3) scenario.prepare_decum => scenario_dict
      4) add_initial_pots => final pot at retirement
      5) MC decum
    """
    # 1) Grow preexisting pots
    init_dict = accumulate_initial_pots(p)

    # 2) scenario accumulate
    scenario_dict = scenario.accumulate(p)

    # 3) scenario prepare_decum
    scenario_dict = scenario.prepare_decum(scenario_dict, p)

    # 4) merge
    pot_dict_init = add_initial_pots(scenario_dict, p, init_dict)

    # 5) Monte Carlo decum
    years_payout = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire

    results = []
    leftover_pots = []
    outcount = 0

    for _ in range(p.num_sims):
        sim_pot = dict(pot_dict_init)  # shallow copy
        total_spend = 0.0
        ran_out = False
        spend_year = p.desired_spend
        T = years_payout[np.random.randint(len(years_payout))]

        for t in range(T):
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)

            needed_net = spend_year
            net_wd, sim_pot = scenario.decumulate_year(
                sim_pot, p, t, needed_net, rand_returns={"eq": eq_r, "bd": bd_r}
            )

            total_spend += present_value(net_wd, t, p.discount_rate)

            infl = np.random.normal(p.inflation_mean, p.inflation_std)
            spend_year *= 1 + infl

            leftover_now = 0.0
            for k in ["eq", "bd", "br_eq", "br_bd", "l3_eq", "l3_bd"]:
                leftover_now += sim_pot.get(k, 0.0)

            if leftover_now <= 0:
                ran_out = True
                break

        results.append(total_spend)
        leftover_pots.append(leftover_now)
        if ran_out:
            outcount += 1

    arr = np.array(results)
    return {
        "prob_runout": outcount / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        # leftover pot
        "p50pot": np.percentile(leftover_pots, 50),
        "all_spend": arr,
        "all_pots": np.array(leftover_pots),
    }


# --------------------------------------------------------
# 8. PLOTTING HELPERS
# --------------------------------------------------------
def plot_distribution(spend_array, scenario_name="Scenario"):
    plt.figure(figsize=(6, 4))
    plt.hist(spend_array, bins=40, alpha=0.7, color="blue")
    plt.title(f"{scenario_name}: Distribution of Total Discounted Spending")
    plt.xlabel("NPV of Spend")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_boxplot(data_dicts, labels, title="Boxplot"):
    """
    data_dicts: list of result dicts
    labels: list of scenario names
    """
    all_data = [res["all_spend"] for res in data_dicts]
    plt.figure(figsize=(7, 4))
    plt.boxplot(all_data, labels=labels)
    plt.title(title)
    plt.ylabel("Discounted Spending")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------
# 9. MAIN (Example Usage)
# --------------------------------------------------------
if __name__ == "__main__":
    p = Params()

    # Instantiate all scenarios
    scenarioA = ScenarioA()
    scenarioB = ScenarioB()
    scenarioC = ScenarioC()
    scenarioD = ScenarioD()
    scenarioE = ScenarioE()
    scenarioF = ScenarioF()

    # Simulate
    resA = simulate_montecarlo(scenarioA, p)
    resB = simulate_montecarlo(scenarioB, p)
    resC = simulate_montecarlo(scenarioC, p)
    resD = simulate_montecarlo(scenarioD, p)
    resE = simulate_montecarlo(scenarioE, p)
    resF = simulate_montecarlo(scenarioF, p)

    # Let's put them together
    results_named = [
        ("Scenario A", resA),
        ("Scenario B", resB),
        ("Scenario C", resC),
        ("Scenario D", resD),
        ("Scenario E", resE),
        ("Scenario F", resF),
    ]

    # Sort by the 'p50pot' (median leftover pot)
    results_sorted = sorted(results_named, key=lambda x: x[1]["p50pot"], reverse=True)

    # Print & plot
    print("Sorted by median leftover pot (p50pot):\n")
    for name, r in results_sorted:
        print(
            f"{name}: p50 leftover = {r['p50pot']:.0f}, run-out = {100*r['prob_runout']:.1f}%, p50 spend = {r['p50']:,.0f}"
        )

    # Box plot (sorted by p50pot)
    plot_boxplot(
        [x[1] for x in results_sorted],
        labels=[x[0] for x in results_sorted],
        title="Scenarios Sorted by p50pot",
    )
