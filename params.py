class Params:
    # Ages
    age_start = 1
    age_transition = 62
    age_retire = 67
    gender = "M"

    # Durations
    years_accum = age_transition - age_start
    years_transition = age_retire - age_transition
    glide_path_years = 20

    # Annual contributions
    annual_contribution = 1

    pension_fee = 0.003
    fund_fee = 0.002

    # Decumulation (glide path) returns
    equity_mean = 0.08
    equity_std = 0.18
    bond_mean = 0.04
    bond_std = 0.05

    # Taxes
    cg_tax_normal = 0.26375  # 26.375% normal CG
    cg_tax_half = 0.26375 / 2  # ~13.1875% for lumpsum
    ruerup_tax = 0.28  # 28% flat on Rürup annuity

    # Rürup annuity
    ruerup_ann_rate = 0.03
    ruerup_dist_fee = 0.015

    # Marginal tax while working => invests refund in scenario B, D
    tax_working = 0.4431

    # Desired net annual spending at retirement
    inflation_mean = 0.02
    inflation_std = 0.01
    desired_spend = 1 * (1.02 ** (years_accum))

    # Number of Monte Carlo runs
    num_sims = 50_000  # for speed in example

    # Discount rate for net present value
    discount_rate = 0.01

    # Preexisting Pots
    initial_rurup = 1    # Convert to an annuity at retirement
    initial_broker = 1   # Follows eq->bd glide path
    initial_broker_bs = 1
    initial_l3 = 1
    initial_l3_bs = 1
    public_pension = 1
