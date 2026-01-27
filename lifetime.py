from __future__ import annotations
from typing import Optional
import numpy as np

# --------------------------------------------------------
# MORTALITY DATA AND HELPER FUNCTIONS
# --------------------------------------------------------

# German mortality data based on the 2018-2020 period life table from the Federal Statistical Office (Destatis)
# Source: https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Tabellen/sterbetafel.html
# Values represent the probability of death within one year at each age (qx)
# These are period rates that don't account for future mortality improvements

# Ages from 67 to 105
AGES = np.arange(67, 106)

# Probability of death at each age (qx) for males
MALE_QX = np.array([
    0.01689,
    0.01861,
    0.02052,
    0.02264,
    0.02501,
    0.02766,
    0.03063,
    0.03396,
    0.03769,
    0.04186,  # 67-76
    0.04651,
    0.05169,
    0.05744,
    0.06382,
    0.07089,
    0.07870,
    0.08731,
    0.09676,
    0.10708,
    0.11831,  # 77-86
    0.13045,
    0.14351,
    0.15747,
    0.17231,
    0.18798,
    0.20443,
    0.22159,
    0.23939,
    0.25775,
    0.27659,  # 87-96
    0.29582,
    0.31536,
    0.33513,
    0.35504,
    0.37500,
    0.39496,
    0.41489,
    0.43478,
    0.45455,  # 97-105
])

# Probability of death at each age (qx) for females
FEMALE_QX = np.array([
    0.00981,
    0.01093,
    0.01220,
    0.01364,
    0.01527,
    0.01712,
    0.01922,
    0.02160,
    0.02430,
    0.02736,  # 67-76
    0.03082,
    0.03473,
    0.03914,
    0.04410,
    0.04967,
    0.05592,
    0.06293,
    0.07078,
    0.07954,
    0.08929,  # 77-86
    0.10008,
    0.11195,
    0.12493,
    0.13903,
    0.15425,
    0.17056,
    0.18792,
    0.20627,
    0.22554,
    0.24565,  # 87-96
    0.26651,
    0.28802,
    0.31007,
    0.33255,
    0.35536,
    0.37839,
    0.40155,
    0.42474,
    0.44787,  # 97-105
])

# Annual mortality improvement factors (estimated)
# These represent how much mortality rates improve each year
# Positive values mean mortality rates decrease over time
MORTALITY_IMPROVEMENT = 0.01  # 1% annual improvement


def get_survival_probabilities(qx: np.ndarray) -> np.ndarray:
    """
    Convert probabilities of death (qx) to survival probabilities.

    Args:
        qx: Array of probabilities of death at each age

    Returns:
        Array of probabilities of surviving from age 67 to each age
    """
    # Convert qx to px (probability of surviving one year)
    px = 1 - qx

    # Calculate cumulative survival probabilities
    survival_probs = np.cumprod(px)

    # Prepend 1.0 for age 67 (100% chance of being alive at starting age)
    return np.concatenate(([1.0], survival_probs[:-1]))


def adjust_for_cohort(qx: np.ndarray, birth_year: int, reference_year: int = 1955) -> np.ndarray:
    """
    Adjust mortality rates for cohort effects.

    Args:
        qx: Array of probabilities of death at each age
        birth_year: Birth year of the individual
        reference_year: Reference birth year for the base mortality table

    Returns:
        Adjusted array of death probabilities
    """
    # Calculate years of improvement
    years_of_improvement = birth_year - reference_year

    # FIXED: Apply compound improvement correctly (positive years should reduce mortality)
    improvement_factor = (1 - MORTALITY_IMPROVEMENT) ** years_of_improvement

    # FIXED: Remove arbitrary 50% mortality reduction cap - let natural improvement work
    adjusted_qx = qx * improvement_factor
    
    # Ensure mortality rates don't go negative or exceed 1
    adjusted_qx = np.clip(adjusted_qx, 0.0001, 1.0)

    return adjusted_qx


def sample_lifetime_from67(
    gender: str = "M", 
    size: int = 10_000, 
    birth_year: Optional[int] = None
) -> np.ndarray:
    """
    Sample random lifetimes starting from age 67.

    Args:
        gender: "M" for male, "F" for female
        size: Number of random samples to generate
        birth_year: Birth year for cohort adjustments (defaults to None, which uses unadjusted rates)

    Returns:
        Array of random ages at death
    """
    # Select appropriate mortality rates based on gender
    if gender == "M":
        qx = MALE_QX.copy()
    else:
        qx = FEMALE_QX.copy()

    # Apply cohort adjustments if birth year is provided
    if birth_year is not None:
        qx = adjust_for_cohort(qx, birth_year)

    # Calculate survival probabilities
    survival_probs = get_survival_probabilities(qx)

    # FIXED: Calculate PDF from the survival curve correctly
    pdf = np.zeros_like(survival_probs)
    pdf[0] = qx[0]  # Probability of dying at exactly age 67
    for i in range(1, len(pdf)):
        # Probability of surviving to age i and then dying in that year
        pdf[i] = survival_probs[i-1] * qx[i]

    # Normalize PDF
    pdf = pdf / pdf.sum()

    # Sample from the distribution
    return np.random.choice(AGES, p=pdf, size=size)


def get_life_expectancy(gender: str = "M", birth_year: Optional[int] = None) -> float:
    """
    Calculate life expectancy at age 67.

    Args:
        gender: "M" for male, "F" for female
        birth_year: Birth year for cohort adjustments

    Returns:
        Expected remaining years of life at age 67
    """
    # Select appropriate mortality rates based on gender
    if gender == "M":
        qx = MALE_QX.copy()
    else:
        qx = FEMALE_QX.copy()

    # Apply cohort adjustments if birth year is provided
    if birth_year is not None:
        qx = adjust_for_cohort(qx, birth_year)

    # Calculate survival probabilities
    survival_probs = get_survival_probabilities(qx)

    # Calculate life expectancy (sum of survival probabilities)
    life_expectancy = survival_probs.sum()

    return life_expectancy
