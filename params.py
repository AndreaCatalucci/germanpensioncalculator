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
    annual_contribution = 6000

    pension_fee = 0.003
    fund_fee = 0.002

    # Historical data parameters

    # Bootstrapping parameters
    bootstrap_period_length = 5  # Length of historical periods to sample

    # Data sources
    equity_series_id = "SP500"  # FRED series ID for S&P 500
    bond_series_id = (
        "GS10"  # FRED series ID for bonds (10-Year Treasury Constant Maturity Rate)
    )
    data_start_year = 1972  # First year to include in historical data (earliest year for WILL5000IND data)

    # Taxes
    cg_tax_normal = 0.26375  # 26.375% normal CG
    cg_tax_half = 0.26375 / 2  # ~13.1875% for lumpsum
    ruerup_tax = 0.28  # 28% flat on Rürup annuity

    # Rürup annuity
    ruerup_ann_rate = 0.0027
    ruerup_dist_fee = 0.015

    # Marginal tax while working => invests refund in scenario B, D
    tax_working = 0.4431

    # Desired net annual spending at retirement
    inflation_mean = 0.02
    inflation_std = 0.01
    desired_spend = 36000 * (1.02 ** (years_accum))

    # Number of Monte Carlo runs
    num_sims = 1_000  # reduced for testing bootstrapping

    # Discount rate for net present value
    discount_rate = 0.01

    # Preexisting Pots
    initial_rurup = 0  # Convert to an annuity at retirement
    initial_broker = 0  # Follows eq->bd glide path
    initial_broker_bs = 0
    initial_l3 = 0
    initial_l3_bs = 0
    public_pension = 1000
