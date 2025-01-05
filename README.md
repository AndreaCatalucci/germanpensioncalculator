# German Pension Calculator

## 1. Purpose
This script simulates various **retirement scenarios** for Germany, involving a **Level 3 (L3) pension**, a **Broker** account or a **Rurüp**. It models:
- **Accumulation** of pots until age 62 (deterministic),
- **Transition** (ages 62–67) with partial shifts from L3 equity to bonds (stochastic returns),
- **Decumulation** from age 67 onward (stochastic lifetimes + annual withdrawals).

By running multiple scenarios, you can see how different strategies (e.g., pure Broker, Rürup + L3) affect **leftover pot** and **spending** distribution in a Monte Carlo simulation.

---

## 2. Installation & Setup (Poetry)

1. **Clone or download** this repository into your local environment.
2. **Install Poetry** if you haven’t already:
   ```bash
   pip install poetry
   ```
3. **Create and activate** a new environment (optional) or rely on Poetry’s venv management.
4. **Install dependencies**:
   ```bash
   poetry install
   ```
   This reads your `pyproject.toml` or `poetry.lock` to install necessary libraries (e.g., NumPy, Matplotlib).

5. **Run the Script**:
   ```bash
   poetry run python main_script.py
   ```
   or
   ```bash
   poetry shell
   python main_script.py
   ```

---

## 3. Key Parameters to Adjust

Open the script (or `params.py`) and look at the **`Params`** class. The most essential attributes you’d likely customize are:

1. **Age-Related**:
   - `age_start` (default 1): Starting “working” or accumulation age. You might set this to 37 or 40 in real use.
   - `age_transition` (62): When the script transitions from deterministic to partial-lumpsum / random returns.
   - `age_retire` (67): Full retirement age. Often the lumpsum or final scenario logic triggers here.

2. **Durations**:
   - `years_accum`, `years_transition`, `glide_path_years` (20 by default).  
     - **`years_accum`**: Derived from `age_transition - age_start`.  
     - **`years_transition`**: Number of random-return years from 62..67 (5 by default).  
     - **`glide_path_years`**: For eq→bond shifts post-retirement or in certain scenarios (20 is just an example).

3. **Economics / Fees**:
   - `annual_contribution`: How much you contribute each year.
   - `pension_fee`, `fund_fee`: The extra fees (annual) L3 or Rürup imposes. If your L3 plan is more expensive, increase them accordingly.
   - `equity_mean`, `equity_std`: Mean/volatility for equity returns.  
   - `bond_mean`, `bond_std`: Mean/volatility for bond returns.

4. **Taxes**:
   - `cg_tax_normal` (default ~26.375%): Normal capital gains tax for the broker.  
   - `cg_tax_half`: Half that rate (~13.1875%) used for L3 lumpsum discount.  
   - `ruerup_tax`: Flat 28% on Rürup annuity.

5. **Decumulation**:
   - `desired_spend`: Annual net spending at retirement (adjusted for inflation).  
   - `inflation_mean`, `inflation_std`: Average inflation rate and volatility.  
   - `num_sims`: Number of Monte Carlo runs.

6. **Preexisting Pots**:
   - `initial_rurup`, `initial_broker`, `initial_broker_bs`, `initial_l3`, `initial_l3_bs`: If you already have some pot built up, set these to realistic amounts. Otherwise, keep them minimal for a simpler model.

---

## 4. Running the Script

1. **Adjust** the parameters in `Params` (or override them) to match your real or hypothetical situation.
2. **Choose** which scenario classes (e.g., `ScenarioBroker`, `ScenarioRurupBroker`, etc.) you want to compare. By default, the main script compares all.
3. **Execute**:
   ```bash
   poetry run python calculator.py
   ```
   You’ll see output like:
   ```
   Scenario Broker: leftover p50=XXX, run-out=YY%, p50 spend=ZZZ
   ...
   ```
   Finally, a Matplotlib box plot compares total discounted spending distributions across scenarios.

---

## 5. Understanding the Output

- **Run-out Probability** (`run-out=YY%`):  
  The fraction of Monte Carlo simulations where the pot failed to meet the desired annual spending at least once.
- **p50 leftover** (`leftover p50=XXX`):  
  Median leftover pot at the end of each simulation. Higher is typically safer.
- **p50 spend** (`p50 spend=ZZZ`):  
  Median of the total net-present-value (NPV) of spending you can achieve under that scenario.  

**Plot**:  
A box plot shows the distribution of discounted spending (NPV) for each scenario. Boxes higher on the y-axis generally indicate more total spending potential.

---

## 6. Modifying or Creating Additional Scenarios

- To add your own **Scenario**:
  1. Create a class that inherits `Scenario`.
  2. Implement `accumulate()` (for the deterministic phase) and `decumulate_year()` (for the retirement phase).  
  3. Optionally, override `transition_year()` if you have a special partial lumpsum or eq→bd shifting logic.

- You can also tweak lumpsum rules or fees in `transition_year()` if your plan allows partial lumpsums or earlier bridging.

---

## 7. Support & Next Steps

- If you want more realistic mortality curves or more complex tax rules, adapt `lifetime/sample_lifetime_from67.py` or the lumpsum computations in your scenario class.
- For advanced analytics (e.g., partial lumpsum across multiple years), revise the logic in `transition_year()` or `decumulate_year()` so that lumpsum is not triggered all at once.
- Consider adjusting `glide_path_years` if you want a slower or faster shift from equity to bonds in the Broker portion after retirement.

---

**Enjoy modeling your retirment strategies** to see how different allocations, fee structures, and lumpsum ages influence your retirement spending and leftover pot under uncertain market conditions!