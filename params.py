class Params:
    # Ages
    age_start: int = 39
    age_transition: int = 62
    age_retire: int = 67
    gender: str = "M"

    # Durations (computed in __init__)
    years_accum: int = 0
    years_transition: int = 0
    glide_path_years: int = 20

    # Annual contributions (realistic for 30-year term)
    annual_contribution: float = 6000  # Reasonable contribution level for 30-year Rürup plan

    pension_fee: float = 0.003
    fund_fee: float = 0.002

    # Historical data parameters

    # Bootstrapping parameters
    bootstrap_period_length: int = 10  # FIXED: Extended from 5 to 10 years for better statistical coverage

    # Data sources
    equity_series_id: str = "WILL5000IND"  # Wilshire 5000 Total Market Index
    bond_series_id: str = (
        "GS10"  # FRED series ID for bonds (10-Year Treasury Constant Maturity Rate)
    )
    data_start_year: int = 1972  # First year to include in historical data (earliest year for WILL5000IND data)

    # Taxes
    cg_tax_normal: float = 0.26375  # 26.375% normal CG
    cg_tax_half: float = 0.26375 / 2  # ~13.1875% for lumpsum
    
    # FIXED: Payout taxation parameters
    ruerup_tax_share: float = 1.0  # 100% of Rürup annuity is taxable
    tax_retirement: float = 0.30   # Estimated marginal tax rate in retirement
    
    # FIXED: Teilfreistellung (Partial Tax Exemption)
    tfs_broker: float = 0.30       # 30% for equity ETFs in broker
    tfs_insurance: float = 0.15    # 15% typical for fund-based insurance (Rürup/L3)

    # FIXED: Added Sparerpauschbetrag (capital gains tax allowance)
    sparerpauschbetrag: float = 1000  # Updated to €1,000 allowance for capital gains (2024/2025)
    
    # FIXED: Basis interest rate for Vorabpauschale (Advance Lump Sum Tax)
    # 2.55% for 2023, 2.29% for 2024. Using conservative 2.5% average.
    basiszins: float = 0.025

    # Rürup annuity (30-year term, reasonable estimation based on German market data)
    # Based on 2023/2024 market study: ~26€/month per €10,000 = ~3.12% annual rate
    # Using conservative estimate with safety margin: 2.8% annual rate
    ruerup_ann_rate: float = 0.028  # 2.8% annual annuity rate for 30-year term
    ruerup_dist_fee: float = 0.015

    # Safety margins for return assumptions
    equity_safety_margin: float = -0.01  # -1% safety margin on equity returns
    bond_safety_margin: float = -0.005   # -0.5% safety margin on bond returns

    # Marginal tax while working => invests refund in scenario B, D
    tax_working: float = 0.42

    # Desired net annual spending at retirement
    inflation_mean: float = 0.02
    inflation_std: float = 0.01
    desired_spend: float = 36000  # FIXED: Desired annual spending in today's purchasing power (no compounding)

    # Number of Monte Carlo runs
    num_sims: int = 5_000  # FIXED: Reduced to 1,000 to prevent excessive overflow warnings during testing

    # Discount rate for net present value
    discount_rate: float = 0.01

    # Preexisting Pots
    initial_rurup: float = 0 # Convert to an annuity at retirement
    initial_broker: float = 0  # Follows eq->bd glide path
    initial_broker_bs: float = 0
    initial_l3: float = 0
    initial_l3_bs: float = 0
    public_pension: float = 1400  # FIXED: Updated to realistic German average (was 1000)
    current_age: int = 0

    def __init__(self) -> None:
        """Initialize Params with computed values."""
        self.years_accum = self.age_transition - self.age_start
        self.years_transition = self.age_retire - self.age_transition
        self.current_age = self.age_start
