import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------
# 1. SHARED PARAMETERS
# --------------------------------------------------------
class Params:
    # Ages
    age_start = 37
    age_retire = 67
    gender = "M"

    # Durations
    years_accum = age_retire - age_start  # e.g. 30
    glide_path_years = 10

    # Annual contributions
    annual_contribution = 12_000

    pension_fee = 0.003
    fund_fee = 0.002

    # Decumulation (glide path) returns
    equity_mean = 0.08
    equity_std = 0.18
    bond_mean = 0.04
    bond_std = 0.05

    # Taxes
    cg_tax_normal = 0.26375  # 26.375% normal CG
    cg_tax_half = 0.26375 / 2  # ~13.1875% for lumpsum at 67
    ruerup_tax = 0.28  # 28% flat on Rürup annuity

    # Rürup annuity
    ruerup_ann_rate = 0.04
    ruerup_dist_fee = 0.015

    # Marginal tax while working => invests refund in scenario B, D
    tax_working = 0.4431

    # Desired net annual spending at retirement
    inflation_mean = 0.02
    inflation_std = 0.01
    desired_spend = 36_000.0 * (1.02 ** (years_accum))

    # Number of Monte Carlo runs
    num_sims = 100_000

    # Discount rate for net present value
    discount_rate = 0.01


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
    draws = np.random.choice(ages, p=pdf, size=size)
    return draws


def shift_equity_to_bonds(eq, eq_bs, bd, bd_bs, fraction, cgt):
    """
    Shift 'fraction' of the equity pot into the bond pot.
    fraction = 0.1 => shift 10% of eq pot => bonds.
    """
    moved = eq * fraction
    bs_moved = eq_bs * fraction

    eq_after = eq - moved
    eq_bs_after = eq_bs - bs_moved

    # Gains = moved - bs_moved => taxed at cgt
    tax_amount = (moved - bs_moved) * cgt
    moved_net = moved - tax_amount

    bd_after = bd + moved_net
    bd_bs_after = bd_bs + moved_net
    return eq_after, eq_bs_after, bd_after, bd_bs_after


def present_value(cf, year, discount_rate):
    """Discount a cash flow 'cf' that occurs 'year' years after retirement=0."""
    return cf / ((1 + discount_rate) ** year)


def solve_gross_for_net(eq, bd, eq_basis, bd_basis, net_needed, cg_tax):
    """
    Solve how much 'gross' must be withdrawn to get 'net_needed' after capital gains tax.
    Remove from bonds first, then from equity.
    Returns (gross_wd, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after).
    """
    total_pot = eq + bd
    if total_pot <= 0:
        return (0, 0, 0, 0, 0, 0)

    net_withdrawn = 0
    gross_withdrawn = 0

    # BONDS FIRST
    bd_after = bd
    eq_after = eq
    bd_bs_after = bd_basis
    eq_bs_after = eq_basis
    still_needed = net_needed

    if bd > 0:
        net_bd = bd - (bd - bd_basis) * cg_tax  # net if we fully withdrew
        portion = min(still_needed, net_bd)
        if portion > 0:
            frac = portion / net_bd  # fraction of the entire bond pot
            gross_bd_withdrawal = frac * bd
            bd_bs_after = bd_basis * (1 - frac)
            bd_after = bd - gross_bd_withdrawal

            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # EQUITY NEXT
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
    Base Scenario. Subclasses should implement:
      - accumulate(params): -> dict of pot states
      - prepare_decum(pot_dict, params): -> transformed pot_dict
      - decumulate_year(pot_dict, params, current_year, needed_net, rand_returns)
        -> (net_withdrawn_this_year, updated_pot_dict)
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
# 4. SCENARIO A
#    Broker Only => eq->bond decum
# --------------------------------------------------------
class ScenarioA(Scenario):
    def accumulate(self, p: Params):
        pot = 0.0
        basis = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee
            pot += ann_contr
            basis += ann_contr
            # optionally grow ann_contr with inflation if you want
            # ann_contr *= 1.02

        # final year's return
        pot *= 1 + p.equity_mean - p.fund_fee

        return {
            "eq": pot,
            "eq_bs": basis,
            "bd": 0.0,
            "bd_bs": 0.0,
        }

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        # SHIFT eq->bond during glide_path_years
        if current_year < p.glide_path_years:
            fraction = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
            )

        # Apply random returns
        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        # Partial withdrawal
        _, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_net, p.cg_tax_normal
        )

        pot_dict["eq"] = eq_after
        pot_dict["bd"] = bd_after
        pot_dict["eq_bs"] = eq_bs_after
        pot_dict["bd_bs"] = bd_bs_after

        return net_, pot_dict


# --------------------------------------------------------
# 5. SCENARIO B
#    Rürup + Broker => eq->bond decum
# --------------------------------------------------------
class ScenarioB(Scenario):
    """
    Accumulation:
      - rpot grows with (equity_mean - fund_fee - pension_fee), plus p.annual_contribution
      - Broker pot grows with (equity_mean - fund_fee),
        plus 'refund' = p.annual_contribution * p.tax_working
      - At retirement => Rürup is converted to a net annuity each year
      - Broker pot => eq->bond decum + partial withdrawal of needed = (desired_spend - net_ann)
    """

    def accumulate(self, p: Params):
        rpot = 0.0
        br = 0.0
        br_bs = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            # Rürup
            rpot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            rpot += ann_contr

            # Broker = invests the "refund"
            refund = ann_contr * p.tax_working
            br *= 1 + p.equity_mean - p.fund_fee
            br += refund
            br_bs += refund
            # ann_contr *= 1.02  # optional inflation growth

        # final year's return
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
        # Convert Rürup pot into net annuity
        rpot = pot_dict["rpot"]
        gross_ann = rpot * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)

        pot_dict["net_ann"] = net_ann
        # We no longer need 'rpot' itself for withdrawals because it's annuitized
        pot_dict.pop("rpot")

        return pot_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        # net_ann from Rürup
        net_ann = pot_dict["net_ann"]
        # Our broker pot => eq + bd
        eq = pot_dict["br_eq"]
        bd = pot_dict["br_bd"]
        eq_bs = pot_dict["br_eq_bs"]
        bd_bs = pot_dict["br_bd_bs"]

        # SHIFT eq->bond
        if current_year < p.glide_path_years:
            fraction = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
            )

        # Apply random returns
        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        # needed from broker pot = desired_spend - net_ann
        needed_broker = max(0, needed_net - net_ann)

        # partial withdrawal from broker
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_broker, p.cg_tax_normal
        )

        # total net = annuity + net from broker
        total_net = net_ann + net_wd

        pot_dict["br_eq"] = eq_after
        pot_dict["br_bd"] = bd_after
        pot_dict["br_eq_bs"] = eq_bs_after
        pot_dict["br_bd_bs"] = bd_bs_after

        return total_net, pot_dict


# --------------------------------------------------------
# 6. SCENARIO C
#    Level 3 => lumpsum half CG => eq->bond decum
# --------------------------------------------------------
class ScenarioC(Scenario):
    def accumulate(self, p: Params):
        pot = 0.0
        bs = 0.0
        ann_contr = p.annual_contribution

        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            pot += ann_contr
            bs += ann_contr
            # ann_contr *= 1.02  # optional

        pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        return {"l3_pot": pot, "l3_bs": bs}

    def prepare_decum(self, pot_dict, p: Params):
        pot = pot_dict["l3_pot"]
        bs = pot_dict["l3_bs"]
        gains = max(0, pot - bs)
        tax_ = gains * p.cg_tax_half
        net_lump = max(0, pot - tax_)

        # that lumpsum => eq pot
        new_dict = {
            "eq": net_lump,
            "eq_bs": net_lump,
            "bd": 0.0,
            "bd_bs": 0.0,
        }
        return new_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        # SHIFT eq->bond
        if current_year < p.glide_path_years:
            fraction = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
            )

        # Apply random returns
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


# --------------------------------------------------------
# 7. SCENARIO D
#    Rürup + L3 lumpsum => eq->bond decum
# --------------------------------------------------------
class ScenarioD(Scenario):
    """
    Accumulate:
      - Rürup pot (rp) => final annuity
      - L3 pot => lumpsum half CG => eq pot
    Decumulate:
      - net annuity from Rürup
      - partial withdrawal from lumpsum eq->bd
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

            # L3 invests the "refund" = ann_contr * p.tax_working
            refund = ann_contr * p.tax_working
            l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3 += refund
            l3_bs += refund
            # ann_contr *= 1.02  # optional

        rp *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        l3 *= 1 + p.equity_mean - p.fund_fee - p.pension_fee

        return {"rp": rp, "l3": l3, "l3_bs": l3_bs}

    def prepare_decum(self, pot_dict, p: Params):
        # 1) Rürup => net annuity
        rp = pot_dict["rp"]
        gross_ann = rp * p.ruerup_ann_rate
        net_ann = gross_ann * (1 - p.ruerup_tax - p.ruerup_dist_fee)

        # 2) lumpsum from L3 => half CG
        l3 = pot_dict["l3"]
        l3_bs = pot_dict["l3_bs"]
        gains = max(0, l3 - l3_bs)
        tax_ = gains * p.cg_tax_half
        net_l3 = max(0, l3 - tax_)

        new_dict = {
            "net_ann": net_ann,
            "eq": net_l3,
            "eq_bs": net_l3,
            "bd": 0.0,
            "bd_bs": 0.0,
        }
        return new_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        net_ann = pot_dict["net_ann"]
        eq = pot_dict["eq"]
        bd = pot_dict["bd"]
        eq_bs = pot_dict["eq_bs"]
        bd_bs = pot_dict["bd_bs"]

        # SHIFT eq->bond
        if current_year < p.glide_path_years:
            fraction = 1.0 / p.glide_path_years
            eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
            )

        # Apply random returns
        eq *= 1 + rand_returns["eq"]
        bd *= 1 + rand_returns["bd"]

        needed_broker = max(0, needed_net - net_ann)
        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq, bd, eq_bs, bd_bs, needed_broker, p.cg_tax_normal
        )
        total_net = net_ann + net_wd

        pot_dict["eq"] = eq_after
        pot_dict["bd"] = bd_after
        pot_dict["eq_bs"] = eq_bs_after
        pot_dict["bd_bs"] = bd_bs_after

        return total_net, pot_dict


# --------------------------------------------------------
# 8. SCENARIO E
#    L3 as annuity => taxed at 17%? (as per your snippet)
# --------------------------------------------------------
class ScenarioE(Scenario):
    """
    This scenario's original code simply compares net annuity to desired spending
    and returns run-out if spend > net_ann in any year.
    For demonstration, we'll replicate that logic.
    """

    def accumulate(self, p: Params):
        # same accumulation as scenarioC, but then fully turned into annuity at 67
        pot = 0.0
        bs = 0.0
        ann_contr = p.annual_contribution
        for _ in range(p.years_accum):
            pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            pot += ann_contr
            bs += ann_contr
        pot *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
        return {"l3_pot": pot, "l3_bs": bs}

    def prepare_decum(self, pot_dict, p: Params):
        # Convert entire pot to an annuity taxed at 17%?
        # Original code snippet: net_ann = pot * ruerup_ann_rate * (1 - p.ruerup_tax*0.17 - p.ruerup_dist_fee)
        pot = pot_dict["l3_pot"]
        # basis = pot_dict["l3_bs"]  # not used
        gross_ann = pot * p.ruerup_ann_rate
        # The snippet is ambiguous about "p.ruerup_tax * 0.17" but let's do exactly the original:
        net_ann = gross_ann * (1 - p.ruerup_tax * 0.17 - p.ruerup_dist_fee)
        return {"net_ann": net_ann}

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        """
        The original code: if spend_year > net_ann, we count it as run-out.
        We'll return net_ann if spend_year <= net_ann, else partial or 0.
        """
        net_ann = pot_dict["net_ann"]
        # There's no pot to withdraw from.
        # If needed_net > net_ann => we effectively "run out" for that year.
        net_payout = net_ann  # every year
        # We won't modify anything in pot_dict, because there's no leftover pot.
        return net_payout, pot_dict


# --------------------------------------------------------
# 9. SCENARIO F
#    50% L3 with 5yr liquidation at 62..66, 50% Broker => eq->bond at 75
# --------------------------------------------------------
class ScenarioF(Scenario):
    """
    Accumulate:
      - L3 pot eq/bd with partial eq->bd from age 62..66
      - Broker pot eq/bd, all eq until 67
    Decumulate from 67:
      - random returns each year
      - shift broker eq->bond from age 72..(72+glide_path_years)
      - partial withdrawal from combined pots
    """

    def accumulate(self, p: Params):
        # We'll track: (L3 eq, L3 bd, L3 eq_bs, L3 bd_bs) and (Broker eq, Broker bd, eq_bs, bd_bs)
        l3_eq = 0.0
        l3_bd = 0.0
        l3_eq_bs = 0.0
        l3_bd_bs = 0.0

        br_eq = 0.0
        br_bd = 0.0
        br_eq_bs = 0.0
        br_bd_bs = 0.0

        annual_l3_perc = 0.5
        ann_contr = p.annual_contribution

        for year in range(p.years_accum):
            current_age = p.age_start + year

            # 1) apply returns
            l3_eq *= 1 + p.equity_mean - p.fund_fee - p.pension_fee
            l3_bd *= 1 + p.bond_mean - p.fund_fee - p.pension_fee

            br_eq *= 1 + p.equity_mean - p.fund_fee
            br_bd *= 1 + p.bond_mean - p.fund_fee

            # 2) add new contributions (split 50/50 to L3 eq and Broker eq)
            l3_contr = ann_contr * annual_l3_perc
            br_contr = ann_contr * (1 - annual_l3_perc)

            l3_eq += l3_contr
            l3_eq_bs += l3_contr
            br_eq += br_contr
            br_eq_bs += br_contr

            # optionally grow future contributions with inflation
            # ann_contr *= 1.02

            # 3) If current_age >= 62 and < 67, shift L3 eq->bond linearly over 5 years
            #    i.e. each year shift 20%
            if 62 <= current_age < 67:
                fraction = 1.0 / 5.0
                l3_bd_before = l3_bd
                l3_eq, l3_eq_bs, l3_bd, l3_bd_bs = shift_equity_to_bonds(
                    l3_eq, l3_eq_bs, l3_bd, l3_bd_bs, fraction, p.cg_tax_half
                )
                delta_bd = l3_bd - l3_bd_before
                l3_bd -= delta_bd * p.ruerup_dist_fee

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
        """
        At age 67 => no lumpsum or annuity creation. We just begin decum.
        The scenario's eq/bd pots are carried forward.
        """
        return pot_dict

    def decumulate_year(self, pot_dict, p, current_year, needed_net, rand_returns):
        """
        1) random returns
        2) shift broker eq->bd after age 72..(72+p.glide_path_years)
        3) combine L3 + broker for partial withdrawal
        4) proportionally re-split eq/bd back into L3 vs. broker
        """
        # read pots
        eq_l3 = pot_dict["l3_eq"]
        bd_l3 = pot_dict["l3_bd"]
        eq_bs_l3 = pot_dict["l3_eq_bs"]
        bd_bs_l3 = pot_dict["l3_bd_bs"]

        eq_br = pot_dict["br_eq"]
        bd_br = pot_dict["br_bd"]
        eq_bs_br = pot_dict["br_eq_bs"]
        bd_bs_br = pot_dict["br_bd_bs"]

        # 1) apply random returns to each pot
        eq_l3 *= 1 + np.random.normal(p.equity_mean, p.equity_std)
        bd_l3 *= 1 + np.random.normal(p.bond_mean, p.bond_std)

        eq_br *= 1 + rand_returns["eq"]
        bd_br *= 1 + rand_returns["bd"]

        # 2) SHIFT broker eq->bond at ages 72..(72 + p.glide_path_years)
        # i.e. a linear shift over p.glide_path_years after age 72
        # We'll see how many years we are past 72:
        retired_age = p.age_retire + current_year  # e.g. 67 + t
        if retired_age >= 72 and (retired_age - 72) < p.glide_path_years:
            fraction = 1.0 / p.glide_path_years
            eq_br, eq_bs_br, bd_br, bd_bs_br = shift_equity_to_bonds(
                eq_br, eq_bs_br, bd_br, bd_bs_br, fraction, p.cg_tax_normal
            )

        # 3) combine L3 + broker for withdrawal
        eq_total = eq_l3 + eq_br
        bd_total = bd_l3 + bd_br
        eq_bs_total = eq_bs_l3 + eq_bs_br
        bd_bs_total = bd_bs_l3 + bd_bs_br

        _, net_wd, eq_after, bd_after, eq_bs_after, bd_bs_after = solve_gross_for_net(
            eq_total, bd_total, eq_bs_total, bd_bs_total, needed_net, p.cg_tax_normal
        )

        # 4) re-split eq_after, bd_after back into L3 vs. broker in proportion
        #    to their share of eq_total and bd_total before the withdrawal
        # eq_l3 was eq_total * ratio_l3, eq_br was eq_total * ratio_br, etc.
        eq_ratio_l3 = eq_l3 / eq_total if eq_total > 0 else 0
        eq_ratio_br = eq_br / eq_total if eq_total > 0 else 0
        bd_ratio_l3 = bd_l3 / bd_total if bd_total > 0 else 0
        bd_ratio_br = bd_br / bd_total if bd_total > 0 else 0

        pot_dict["l3_eq"] = eq_after * eq_ratio_l3
        pot_dict["br_eq"] = eq_after * eq_ratio_br
        pot_dict["l3_bd"] = bd_after * bd_ratio_l3
        pot_dict["br_bd"] = bd_after * bd_ratio_br

        pot_dict["l3_eq_bs"] = eq_bs_after * eq_ratio_l3
        pot_dict["br_eq_bs"] = eq_bs_after * eq_ratio_br
        pot_dict["l3_bd_bs"] = bd_bs_after * bd_ratio_l3
        pot_dict["br_bd_bs"] = bd_bs_after * bd_ratio_br

        return net_wd, pot_dict


# --------------------------------------------------------
# 10. MONTE CARLO RUNNER
# --------------------------------------------------------
def simulate_montecarlo(scenario: Scenario, p: Params):
    # 1) Accumulate
    pot_dict_accum = scenario.accumulate(p)
    # 2) Prepare decum (e.g. lumpsum, annuity, etc.)
    pot_dict_init = scenario.prepare_decum(pot_dict_accum, p)

    # 3) For each simulation => random lifetime
    years_payout = sample_lifetime_from67(p.gender, p.num_sims) - p.age_retire

    results = []
    leftover_pots = []
    outcount = 0

    for i in range(p.num_sims):
        sim_pot = dict(pot_dict_init)  # shallow copy
        total_spend = 0.0
        ran_out = False
        spend_year = p.desired_spend
        T = years_payout[i]

        for t in range(T):
            # Random returns
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)

            needed_net = spend_year
            net_withdrawn, sim_pot = scenario.decumulate_year(
                sim_pot,
                p,
                current_year=t,
                needed_net=needed_net,
                rand_returns={"eq": eq_r, "bd": bd_r},
            )

            # discount
            total_spend += present_value(net_withdrawn, t, p.discount_rate)

            # inflation
            infl = np.random.normal(p.inflation_mean, p.inflation_std)
            spend_year *= 1 + infl

            # check run-out
            # Different scenarios store pot differently, but let's guess eq+bd is the main leftover
            leftover_now = 0.0
            if "eq" in sim_pot:
                leftover_now += sim_pot["eq"]
            if "bd" in sim_pot:
                leftover_now += sim_pot["bd"]
            if "br_eq" in sim_pot:
                leftover_now += sim_pot["br_eq"]
            if "br_bd" in sim_pot:
                leftover_now += sim_pot["br_bd"]
            if "l3_eq" in sim_pot:
                leftover_now += sim_pot["l3_eq"]
            if "l3_bd" in sim_pot:
                leftover_now += sim_pot["l3_bd"]

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
        "p50pot": np.percentile(leftover_pots, 50),
        "all_spend": arr,
        "all_pots": np.array(leftover_pots),
    }


# --------------------------------------------------------
# 11. PLOTTING HELPERS
# --------------------------------------------------------
def plot_distribution(spend_array, scenario_name="Scenario"):
    plt.figure(figsize=(6, 4))
    plt.hist(spend_array, bins=50, alpha=0.7, color="blue")
    plt.title(f"{scenario_name}: Distribution of Total Discounted Spending")
    plt.xlabel("NPV of Spend")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_boxplot(data_dicts, labels):
    all_data = [res["all_spend"] for res in data_dicts]
    plt.figure(figsize=(6, 4))
    plt.boxplot(all_data, labels=labels)
    plt.title("Boxplot of NPV of Spending by Scenario")
    plt.ylabel("NPV of Spending")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------
# 12. MAIN (Example Usage)
# --------------------------------------------------------
if __name__ == "__main__":
    p = Params()

    # Scenario A
    scenarioA = ScenarioA()
    resA = simulate_montecarlo(scenarioA, p)
    print("\nScenario A (Broker Only, eq->bond decum):")
    print(f"Run-out prob: {resA['prob_runout']*100:.1f}%")
    print(
        f"10%: {resA['p10']:,.0f}, median: {resA['p50']:,.0f}, 90%: {resA['p90']:,.0f}, leftover pot: {resA['p50pot']:,.0f}"
    )

    # Scenario B
    scenarioB = ScenarioB()
    resB = simulate_montecarlo(scenarioB, p)
    print("\nScenario B (Rürup + Broker, eq->bond decum):")
    print(f"Run-out prob: {resB['prob_runout']*100:.1f}%")
    print(
        f"10%: {resB['p10']:,.0f}, median: {resB['p50']:,.0f}, 90%: {resB['p90']:,.0f}, leftover pot: {resB['p50pot']:,.0f}"
    )

    # Scenario C
    scenarioC = ScenarioC()
    resC = simulate_montecarlo(scenarioC, p)
    print("\nScenario C (Level3 lumpsum half CG => eq->bond decum):")
    print(f"Run-out prob: {resC['prob_runout']*100:.1f}%")
    print(
        f"10%: {resC['p10']:,.0f}, median: {resC['p50']:,.0f}, 90%: {resC['p90']:,.0f}, leftover pot: {resC['p50pot']:,.0f}"
    )

    # Scenario D
    scenarioD = ScenarioD()
    resD = simulate_montecarlo(scenarioD, p)
    print("\nScenario D (Rürup + L3 lumpsum => eq->bond decum):")
    print(f"Run-out prob: {resD['prob_runout']*100:.1f}%")
    print(
        f"10%: {resD['p10']:,.0f}, median: {resD['p50']:,.0f}, 90%: {resD['p90']:,.0f}, leftover pot: {resD['p50pot']:,.0f}"
    )

    # Scenario E
    scenarioE = ScenarioE()
    resE = simulate_montecarlo(scenarioE, p)
    print("\nScenario E (L3 annuity at 17% tax):")
    print(f"Run-out prob: {resE['prob_runout']*100:.1f}%")
    print(
        f"10%: {resE['p10']:,.0f}, median: {resE['p50']:,.0f}, 90%: {resE['p90']:,.0f}, leftover pot: {resE['p50pot']:,.0f}"
    )

    # Scenario F
    scenarioF = ScenarioF()
    resF = simulate_montecarlo(scenarioF, p)
    print(
        "\nScenario F (50% L3 w/5yr liquidation at 62, 50% Broker => shift eq->bd at 75):"
    )
    print(f"Run-out prob: {resF['prob_runout']*100:.1f}%")
    print(
        f"10%: {resF['p10']:,.0f}, median: {resF['p50']:,.0f}, 90%: {resF['p90']:,.0f}, leftover pot: {resF['p50pot']:,.0f}"
    )

    # Example Plots
    # Compare a few scenarios in a box plot:
    sorted_res = sorted(
        [
            (resA, "Broker Only"),
            (resB, "Rürup + Broker"),
            (resC, "L3 Lump Sum"),
            (resD, "Rürup + L3 Lump Sum"),
            (resE, "L3 Annutiy"),
            (resF, "50% L3 50% Broker"),
        ],
        key=lambda t: t[0]["p50"],
        reversed=True,
    )
    plot_boxplot([e[0] for e in sorted_res], labels=[e[1] for e in sorted_res])
