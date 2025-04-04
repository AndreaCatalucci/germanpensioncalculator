import numpy as np


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
