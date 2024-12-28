import numpy as np


# --------------------------------------------------------
# SHARED PARAMETERS
# --------------------------------------------------------
class Params:
    # Ages
    age_start = 38
    age_retire = 67
    age_death_you = 85

    # Durations
    years_accum = age_retire - age_start  # e.g. 29
    years_payout = age_death_you - age_retire  # e.g. 18
    glide_path_years = 20  # number of years to shift from equity to bonds post-67

    # Annual contributions
    annual_contribution = 16000

    pension_fee = 0.003
    fund_fee = 0.002

    # Decumulation (glide path) returns
    equity_mean = 0.08
    equity_std = 0.18
    bond_mean = 0.04
    bond_std = 0.05

    # Accumulation returns
    r_broker_accum = equity_mean - fund_fee
    r_ruerup_accum = equity_mean - pension_fee - fund_fee
    r_level3_accum = equity_mean - pension_fee - fund_fee

    # Taxes
    cg_tax_normal = 0.26375  # 26.375% normal CG
    cg_tax_half = 0.26375 / 2  # ~13.1875% for Level3 lumpsum at 67
    ruerup_tax = 0.28  # 28% flat on Rürup annuity

    # Rürup annuity rate
    ruerup_ann_rate = 0.04

    # Marginal tax while working => invests refund in scenario B,D
    tax_working = 0.4431

    # Desired net annual spending
    inflation = 0.02
    desired_spend = 48000.0 * ((1 + inflation) ** years_accum)

    # Number of Monte Carlo runs
    num_sims = 1000

    # Discount rate for net present value
    discount_rate = 0.00


# --------------------------------------------------------
# HELPER: SHIFT eq->bonds
# --------------------------------------------------------
def shift_equity_to_bonds(eq, eq_bs, bd, bd_bs, fraction, cgt):
    """
    Shift 'fraction' of the equity pot into bonds pot.
    fraction = 0.1 => shift 10% of eq pot => bonds.
    """
    moved = eq * fraction
    bs_moved = eq_bs * fraction
    eq_after = eq - moved
    moved_net = moved - (moved - bs_moved) * cgt
    bd_after = bd + moved_net
    return eq_after, eq_bs - bs_moved, bd_after, bd_bs + moved_net


def present_value(cf, year, discount_rate):
    """Discount a cash flow 'cf' that occurs 'year' years after retirement=0."""
    return cf / ((1 + discount_rate) ** year)


# Solve how much 'gross' we must withdraw to get 'net_needed' after capital gains tax.
def solve_gross_for_net(eq, bd, eq_basis, bd_basis, net_needed, cg_tax):
    """
    Combined pot = eq + bd.
    fraction_gains = ( (eq + bd) - (eq_basis + bd_basis) ) / (eq+bd)
    Gains portion of 'gross' = gross * fraction_gains
    Tax = cg_tax * that gains
    net = gross - tax
    Return (gross, actual_net, eq_after, bd_after, eq_basis_after, bd_basis_after)
    """
    # we remove from bonds first
    bd_after = bd
    net_withdrawal = 0
    gross_withdrawal = 0
    eq_after = eq
    eq_basis_after = eq_basis
    bd_basis_after = bd_basis
    if bd > 0:
        net_bd = bd - (bd - bd_basis) * cg_tax
        net_bd_withdrawal = min(net_needed, net_bd)
        net_needed -= net_bd_withdrawal
        gross_bd_withdrawal = net_bd_withdrawal * (1 + cg_tax)
        bd_after = bd - gross_bd_withdrawal
        bd_basis_after = bd_basis_after * (gross_bd_withdrawal / bd)
        net_withdrawal += net_bd_withdrawal
        gross_withdrawal += gross_bd_withdrawal
    if eq > 0:
        net_eq = eq - (eq - eq_basis) * cg_tax
        net_eq_withdrawal = min(net_needed, net_eq)
        net_needed -= net_eq_withdrawal
        gross_eq_withdrawal = net_eq_withdrawal * (1 + cg_tax)
        eq_after = eq - gross_eq_withdrawal
        eq_basis_after = eq_basis_after * (gross_eq_withdrawal / eq)
        net_withdrawal += net_eq_withdrawal
        gross_withdrawal += gross_eq_withdrawal
    return (
        gross_withdrawal,
        net_withdrawal,
        eq_after,
        bd_after,
        eq_basis_after,
        bd_basis_after,
    )


# --------------------------------------------------------
# ACCUMULATIONS
# --------------------------------------------------------
def scenarioA_accum(p: Params):
    pot = 0.0
    basis = 0.0
    for _ in range(p.years_accum):
        pot *= 1 + p.r_broker_accum
        pot += p.annual_contribution
        basis += p.annual_contribution
    pot *= 1 + p.r_broker_accum
    return pot, basis


def scenarioB_accum(p: Params):
    rpot = 0.0
    br = 0.0
    br_bs = 0.0
    for _ in range(p.years_accum):
        rpot *= 1 + p.r_ruerup_accum
        rpot += p.annual_contribution

        ref_ = p.annual_contribution * p.tax_working
        br *= 1 + p.r_broker_accum
        br += ref_
        br_bs += ref_
    rpot *= 1 + p.r_ruerup_accum
    br *= 1 + p.r_broker_accum
    return rpot, br, br_bs


def scenarioC_accum(p: Params):
    pot = 0.0
    bs = 0.0
    for _ in range(p.years_accum):
        pot *= 1 + p.r_level3_accum
        pot += p.annual_contribution
        bs += p.annual_contribution
    pot *= 1 + p.r_level3_accum
    return pot, bs


def scenarioD_accum(p: Params):
    rp = 0.0
    l3 = 0.0
    l3_bs = 0.0
    for _ in range(p.years_accum):
        rp *= 1 + p.r_ruerup_accum
        rp += p.annual_contribution

        ref_ = p.annual_contribution * p.tax_working
        l3 *= 1 + p.r_level3_accum
        l3 += ref_
        l3_bs += ref_
    rp *= 1 + p.r_ruerup_accum
    l3 *= 1 + p.r_level3_accum
    return rp, l3, l3_bs


# --------------------------------------------------------
# SCENARIO A: Monte Carlo with eq pot only
# --------------------------------------------------------
def scenarioA_montecarlo(p: Params):
    # Accumulate
    pot, bs = scenarioA_accum(p)
    # At 67, entire pot is in eq? We can start eq=pot, eq_bs=bs, bd=0, bd_bs=0

    results = []
    result_pot = []
    outcount = 0
    for _ in range(p.num_sims):
        total_net = 0.0
        total_spend = 0.0
        ran_out = False
        eq = pot
        eq_bs = bs
        bd = 0.0
        bd_bs = 0.0
        for t in range(p.years_payout):
            if t < p.glide_path_years:
                fraction = 1.0 / p.glide_path_years
                eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                    eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
                )
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)
            eq *= 1 + eq_r
            bd *= 1 + bd_r

            # partial withdrawal
            gross, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = (
                solve_gross_for_net(
                    eq, bd, eq_bs, bd_bs, p.desired_spend, p.cg_tax_normal
                )
            )

            eq = eq_after
            bd = bd_after
            eq_bs = eq_bs_after
            bd_bs = bd_bs_after

            total_spend += present_value(net_, t, p.discount_rate)

            if eq + bd <= 0:
                eq = 0
                bd = 0
                ran_out = True

        results.append(total_spend)
        result_pot.append(eq + bd)
        if ran_out:
            outcount += 1
    arr = np.array(results)
    return {
        "prob_runout": outcount / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p50pot": np.percentile(result_pot, 50),
    }


# --------------------------------------------------------
# SCENARIO B: Rürup + Broker with eq->bond decum
# --------------------------------------------------------
def scenarioB_montecarlo(p: Params):
    rp, br, br_bs = scenarioB_accum(p)
    # Rürup => annuity net each year
    gross_ann = rp * p.ruerup_ann_rate
    net_ann = gross_ann * (1 - p.ruerup_tax)

    # The broker pot => eq=br, eq_bs=br_bs, bd=0, bd_bs=0
    results = []
    result_pot = []
    outc = 0
    for _ in range(p.num_sims):
        eq = br
        eq_bs = br_bs
        bd = 0.0
        bd_bs = 0.0

        total_spend = 0.0
        ran_out = False

        for t in range(p.years_payout):
            # SHIFT eq->bond
            if t < p.glide_path_years:
                fraction = 1.0 / p.glide_path_years
                eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                    eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
                )

            # random returns eq & bd
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)
            eq *= 1 + eq_r
            bd *= 1 + bd_r

            needed = p.desired_spend - net_ann
            if needed < 0:
                needed = 0
            # partial withdrawal
            gross, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = (
                solve_gross_for_net(eq, bd, eq_bs, bd_bs, needed, p.cg_tax_normal)
            )

            eq = eq_after
            bd = bd_after
            eq_bs = eq_bs_after
            bd_bs = bd_bs_after

            total_net = net_ann + net_
            total_spend += present_value(total_net, t, p.discount_rate)

            if eq + bd <= 0:
                eq = 0
                bd = 0
                ran_out = True

        results.append(total_spend)
        result_pot.append(eq + bd)
        if ran_out:
            outc += 1

    arr = np.array(results)
    return {
        "prob_runout": outc / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p50pot": np.percentile(result_pot, 50),
    }


# --------------------------------------------------------
# SCENARIO C: Level 3 => lumpsum half CG => eq->bond decum
# --------------------------------------------------------
def scenarioC_montecarlo(p: Params):
    pot, bs = scenarioC_accum(p)
    # lumpsum at 67 => half CG
    gains = max(0, pot - bs)
    tax_ = gains * p.cg_tax_half
    net_lump = pot - tax_
    if net_lump < 0:
        net_lump = 0

    # That net lumpsum => new decum pot eq= net_lump, eq_bs= net_lump
    # Then eq->bond + partial withdrawal
    results = []
    result_pot = []
    outc = 0
    for _ in range(p.num_sims):
        eq = net_lump
        eq_bs = net_lump
        bd = 0.0
        bd_bs = 0.0

        total_spend = 0.0
        ran_out = False

        for t in range(p.years_payout):
            # SHIFT
            if t < p.glide_path_years:
                fraction = 1.0 / p.glide_path_years
                eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                    eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
                )

            # random returns
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)
            eq *= 1 + eq_r
            bd *= 1 + bd_r

            # partial withdrawal
            gross, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = (
                solve_gross_for_net(
                    eq, bd, eq_bs, bd_bs, p.desired_spend, p.cg_tax_normal
                )
            )
            eq = eq_after
            bd = bd_after
            eq_bs = eq_bs_after
            bd_bs = bd_bs_after

            total_spend += present_value(net_, t, p.discount_rate)

            result_pot.append(eq + bd)
            if eq + bd <= 0:
                eq = 0
                bd = 0
                ran_out = True

        results.append(total_spend)
        if ran_out:
            outc += 1

    arr = np.array(results)
    return {
        "prob_runout": outc / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p50pot": np.percentile(result_pot, 50),
    }


# --------------------------------------------------------
# SCENARIO D: Rürup + L3 => lumpsum half CG => eq->bond decum
# --------------------------------------------------------
def scenarioD_montecarlo(p: Params):
    rp, l3, l3_bs = scenarioD_accum(p)
    # 1) Rürup => annuity net each year
    gross_annu = rp * p.ruerup_ann_rate
    net_annu = gross_annu * (1 - p.ruerup_tax)

    # 2) lumpsum L3 => half CG
    gains = max(0, l3 - l3_bs)
    tax_ = gains * p.cg_tax_half
    net_l3 = l3 - tax_
    if net_l3 < 0:
        net_l3 = 0

    # That net => eq pot
    # Then partial decum each year => net needed= p.desired_spend - net_annu
    results = []
    result_pot = []
    outcount = 0

    for _ in range(p.num_sims):
        eq = net_l3
        eq_bs = net_l3
        bd = 0.0
        bd_bs = 0.0

        total_spend = 0.0
        ran_out = False

        for t in range(p.years_payout):
            # SHIFT eq->bond
            if t < p.glide_path_years:
                fraction = 1.0 / p.glide_path_years
                eq, eq_bs, bd, bd_bs = shift_equity_to_bonds(
                    eq, eq_bs, bd, bd_bs, fraction, p.cg_tax_normal
                )

            # random returns
            eq_r = np.random.normal(p.equity_mean, p.equity_std)
            bd_r = np.random.normal(p.bond_mean, p.bond_std)
            eq *= 1 + eq_r
            bd *= 1 + bd_r

            needed = p.desired_spend - net_annu
            if needed < 0:
                needed = 0

            gross, net_, eq_after, bd_after, eq_bs_after, bd_bs_after = (
                solve_gross_for_net(eq, bd, eq_bs, bd_bs, needed, p.cg_tax_normal)
            )

            eq = eq_after
            bd = bd_after
            eq_bs = eq_bs_after
            bd_bs = bd_bs_after

            total_net = net_annu + net_
            total_spend += present_value(total_net, t, p.discount_rate)
            result_pot.append(present_value(eq + bd, t, p.discount_rate))

            if eq + bd <= 0:
                eq = 0
                bd = 0
                ran_out = True

        results.append(total_spend)
        if ran_out:
            outcount += 1

    arr = np.array(results)
    return {
        "prob_runout": outcount / p.num_sims,
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p50pot": np.percentile(result_pot, 50),
    }


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    p = Params()

    resA = scenarioA_montecarlo(p)
    resB = scenarioB_montecarlo(p)
    resC = scenarioC_montecarlo(p)
    resD = scenarioD_montecarlo(p)

    print("\nScenario A (Broker Only, eq->bond decum):")
    print(f"Run-out prob: {resA['prob_runout']*100:.1f}%")
    print(
        f"10%: {resA['p10']:,.0f}, median: {resA['p50']:,.0f}, 90%: {resA['p90']:,.0f}, pot: {resA['p50pot']}"
    )

    print("\nScenario B (Rürup + Broker, eq->bond decum):")
    print(f"Run-out prob: {resB['prob_runout']*100:.1f}%")
    print(
        f"10%: {resB['p10']:,.0f}, median: {resB['p50']:,.0f}, 90%: {resB['p90']:,.0f}, pot: {resB['p50pot']}"
    )

    print("\nScenario C (Level3 lumpsum half CG => eq->bond decum):")
    print(f"Run-out prob: {resC['prob_runout']*100:.1f}%")
    print(
        f"10%: {resC['p10']:,.0f}, median: {resC['p50']:,.0f}, 90%: {resC['p90']:,.0f}, pot: {resC['p50pot']}"
    )

    print("\nScenario D (Rürup + L3 lumpsum => eq->bond decum):")
    print(f"Run-out prob: {resD['prob_runout']*100:.1f}%")
    print(
        f"10%: {resD['p10']:,.0f}, median: {resD['p50']:,.0f}, 90%: {resD['p90']:,.0f}, pot: {resD['p50pot']}"
    )
