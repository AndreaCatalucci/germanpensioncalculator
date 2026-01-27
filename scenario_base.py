from __future__ import annotations
from dataclasses import dataclass
import math
import logging
from typing import Union, Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from params import Params

# Configure logging for numerical stability issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Realistic bounds for German retirement planning
MAX_PORTFOLIO_VALUE = 1e9  # 1 billion euros - increased to prevent premature overflow
MIN_PORTFOLIO_VALUE = -1e6  # -1 million euros - reasonable debt limit
MAX_RETURN_RATE = 1.0  # 100% maximum return (prevents extreme overflow while allowing synthetic outliers)
MIN_RETURN_RATE = -0.50  # -50% minimum return (allows for historical crashes)
MAX_AGE = 120  # Maximum realistic age
MIN_AGE = 18   # Minimum working age
MAX_CONTRIBUTION = 1e6  # 1 million euro maximum annual contribution
MIN_CONTRIBUTION = 0    # Minimum contribution

class NumericalStabilityError(Exception):
    """Custom exception for numerical stability issues"""
    pass

class ValidationError(Exception):
    """Custom exception for parameter validation failures"""
    pass

def validate_finite(value: float, name: str = "value") -> float:
    """Validate that a value is finite and return it"""
    if not math.isfinite(value):
        error_msg = f"{name} must be finite, got {value}"
        logger.error(error_msg)
        raise NumericalStabilityError(error_msg)
    return value

def validate_age(age: Union[int, float], name: str = "age") -> float:
    """Validate age is within realistic bounds"""
    age = float(age)
    validate_finite(age, name)
    if not (MIN_AGE <= age <= MAX_AGE):
        error_msg = f"{name} must be between {MIN_AGE} and {MAX_AGE}, got {age}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    return age

def validate_return_rate(rate: float, name: str = "return_rate") -> float:
    """Validate return rate is within reasonable bounds"""
    validate_finite(rate, name)
    if not (MIN_RETURN_RATE <= rate <= MAX_RETURN_RATE):
        error_msg = f"{name} must be between {MIN_RETURN_RATE:.1%} and {MAX_RETURN_RATE:.1%}, got {rate:.1%}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    return rate

def validate_positive_value(value: float, name: str = "value", allow_zero: bool = True) -> float:
    """Validate that a value is positive (or zero if allowed)"""
    validate_finite(value, name)
    min_val = 0.0 if allow_zero else 1e-10
    if value < min_val:
        error_msg = f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    return value

def validate_portfolio_value(value: float, name: str = "portfolio_value") -> float:
    """Validate portfolio value is within reasonable bounds"""
    validate_finite(value, name)
    if not (MIN_PORTFOLIO_VALUE <= value <= MAX_PORTFOLIO_VALUE):
        error_msg = f"{name} must be between {MIN_PORTFOLIO_VALUE:,.0f} and {MAX_PORTFOLIO_VALUE:,.0f}, got {value:,.0f}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    return value

def validate_percentage(value: float, name: str = "percentage", max_percent: float = 1.0) -> float:
    """Validate percentage is within bounds (0-100% by default)"""
    validate_finite(value, name)
    if not (0.0 <= value <= max_percent):
        error_msg = f"{name} must be between 0% and {max_percent:.1%}, got {value:.1%}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    return value

def safe_divide(numerator: float, denominator: float, name: str = "division") -> float:
    """Safely divide with validation"""
    validate_finite(numerator, f"{name}_numerator")
    validate_finite(denominator, f"{name}_denominator")
    
    if abs(denominator) < 1e-10:
        error_msg = f"Division by zero or near-zero in {name}: {denominator}"
        logger.error(error_msg)
        raise NumericalStabilityError(error_msg)
    
    result = numerator / denominator
    validate_finite(result, f"{name}_result")
    return result

def cap_value(value: float, max_val: float = MAX_PORTFOLIO_VALUE) -> float:
    """Cap a value to prevent overflow while maintaining sign"""
    if not math.isfinite(value):
        logger.warning(f"Non-finite value encountered: {value}, returning 0.0")
        return 0.0
    if abs(value) > max_val:
        capped = max_val if value > 0 else -max_val
        logger.warning(f"Value {value:,.0f} capped to {capped:,.0f}")
        return capped
    return value

def safe_multiply(value: float, multiplier: float, max_result: float = MAX_PORTFOLIO_VALUE) -> float:
    """Safely multiply values with overflow protection"""
    if not math.isfinite(value) or not math.isfinite(multiplier):
        logger.warning(f"Non-finite values in multiplication: {value} * {multiplier}, returning 0.0")
        return 0.0
    if multiplier == 0:
        return 0.0
    if abs(value) > max_result / abs(multiplier):
        result = max_result if (value > 0) == (multiplier > 0) else -max_result
        logger.warning(f"Multiplication overflow prevented: {value:,.0f} * {multiplier:.4f} capped to {result:,.0f}")
        return result
    result = value * multiplier
    return cap_value(result, max_result)

def safe_add(*values: float, max_result: float = MAX_PORTFOLIO_VALUE) -> float:
    """Safely add multiple values with overflow protection"""
    total = 0.0
    for i, value in enumerate(values):
        if not math.isfinite(value):
            logger.warning(f"Non-finite value in addition at index {i}: {value}, skipping")
            continue
        total += value
        if abs(total) > max_result:
            result = max_result if total > 0 else -max_result
            logger.warning(f"Addition overflow prevented: sum capped to {result:,.0f}")
            return result
    return cap_value(total, max_result)

def safe_subtract(minuend: float, subtrahend: float, max_result: float = MAX_PORTFOLIO_VALUE) -> float:
    """Safely subtract values with overflow protection"""
    if not math.isfinite(minuend) or not math.isfinite(subtrahend):
        logger.warning(f"Non-finite values in subtraction: {minuend} - {subtrahend}, returning 0.0")
        return 0.0
    result = minuend - subtrahend
    return cap_value(result, max_result)

def validate_calculation_result(result: float, operation: str, inputs: Optional[Dict[str, Any]] = None) -> float:
    """Validate the result of a calculation and log issues"""
    if not math.isfinite(result):
        input_info = f" with inputs {inputs}" if inputs else ""
        error_msg = f"Non-finite result from {operation}{input_info}: {result}"
        logger.error(error_msg)
        raise NumericalStabilityError(error_msg)
    
    if abs(result) > MAX_PORTFOLIO_VALUE:
        input_info = f" with inputs {inputs}" if inputs else ""
        logger.warning(f"Large result from {operation}{input_info}: {result:,.0f}")
    
    return result


@dataclass
class Pot:
    rurup: float = 0.0

    br_eq: float = 0.0
    br_eq_bs: float = 0.0
    br_bd: float = 0.0
    br_bd_bs: float = 0.0

    l3_eq: float = 0.0
    l3_eq_bs: float = 0.0

    def __post_init__(self):
        """Ensure all fields are scalars to prevent array handling issues."""
        self.rurup = float(self.rurup)
        self.br_eq = float(self.br_eq)
        self.br_eq_bs = float(self.br_eq_bs)
        self.br_bd = float(self.br_bd)
        self.br_bd_bs = float(self.br_bd_bs)
        self.l3_eq = float(self.l3_eq)
        self.l3_eq_bs = float(self.l3_eq_bs)

    def leftover(self) -> float:
        return safe_add(self.br_eq, self.br_bd, self.l3_eq)


@dataclass
class Withdrawal:
    gross_withdrawn: float
    net_withdrawn: float


def present_value(cf: float, year: int, discount_rate: float) -> float:
    # Add numerical stability checks
    safe_discount_rate = max(0.001, discount_rate)  # Minimum 0.1% discount rate
    safe_year = max(0, year)  # Ensure non-negative year
    return cf / ((1 + safe_discount_rate) ** safe_year)


def withdraw(
    pot: Pot,
    net_needed: float,
    cg_tax: float,
    sparerpauschbetrag: float = 1000,
    tfs_broker: float = 0.30,
) -> Withdrawal:
    total_pot = safe_add(pot.br_bd, pot.br_eq, pot.l3_eq)
    if total_pot <= 0:
        return Withdrawal(0.0, 0.0)

    net_withdrawn: float = 0.0
    gross_withdrawn: float = 0.0
    still_needed: float = net_needed

    # FIXED: Track remaining Sparerpauschbetrag allowance for the year
    remaining_allowance: float = sparerpauschbetrag
    
    # BOND FIRST
    if pot.br_bd > 0:
        gross_gains_bd: float = max(0.0, pot.br_bd - pot.br_bd_bs)
        allowance_used_bd: float = min(remaining_allowance, gross_gains_bd)
        remaining_allowance -= allowance_used_bd
        
        # Apply tax only to gains above allowance, considering Teilfreistellung
        # Note: TFS usually applies to equity, but bonds in broker also have basis tracking here.
        # We focus TFS on equity below. For bonds, we use full taxation.
        taxable_gains_bd: float = max(0.0, gross_gains_bd - allowance_used_bd)
        net_bd: float = pot.br_bd - taxable_gains_bd * cg_tax
        
        portion: float = min(still_needed, net_bd)
        if portion > 0:
            # FIXED: Calculate fraction based on gross amount for proper proportional withdrawal
            frac: float = portion / net_bd if net_bd > 0 else 0.0
            gross_bd_withdrawal: float = frac * pot.br_bd
            # FIXED: Proportional basis reduction
            pot.br_bd_bs = pot.br_bd_bs * (1 - frac)
            pot.br_bd -= gross_bd_withdrawal
            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # EQUITY NEXT
    if still_needed > 0 and pot.br_eq > 0:
        gross_gains_eq: float = max(0.0, pot.br_eq - pot.br_eq_bs)
        allowance_used_eq: float = min(remaining_allowance, gross_gains_eq)
        
        # Apply tax only to gains above remaining allowance WITH Teilfreistellung
        # TFS: Only (1 - tfs_broker) of gains are taxable
        taxable_gains_eq_pre_tfs: float = max(0.0, gross_gains_eq - allowance_used_eq)
        taxable_gains_eq: float = taxable_gains_eq_pre_tfs * (1 - tfs_broker)
        net_eq: float = pot.br_eq - taxable_gains_eq * cg_tax
        
        portion2: float = min(still_needed, net_eq)
        if portion2 > 0:
            # FIXED: Calculate fraction based on net amount with safety check
            frac2: float = portion2 / net_eq if net_eq > 0 else 0.0
            gross_eq_withdrawal: float = frac2 * pot.br_eq
            # FIXED: Proportional basis reduction
            pot.br_eq_bs = pot.br_eq_bs * (1 - frac2)
            pot.br_eq = safe_subtract(pot.br_eq, gross_eq_withdrawal)
            net_withdrawn += portion2
            gross_withdrawn += gross_eq_withdrawal
            still_needed -= portion2

    return Withdrawal(gross_withdrawn, net_withdrawn)


def shift_equity_to_bonds(pot: Pot, fraction: float, cgt: float) -> None:
    moved = pot.br_eq * fraction
    bs_moved = pot.br_eq_bs * fraction
    pot.br_eq -= moved
    pot.br_eq_bs -= bs_moved
    gains = moved - bs_moved
    tax_amount = gains * cgt
    moved_net = moved - tax_amount
    pot.br_bd += moved_net
    pot.br_bd_bs += moved_net


class Scenario:
    def __init__(self, p: Params) -> None:
        self.params = p
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate all input parameters for numerical stability"""
        try:
            # Age validation
            if hasattr(self.params, 'age_start'):
                validate_age(self.params.age_start, "age_start")
            if hasattr(self.params, 'age_retire'):
                validate_age(self.params.age_retire, "age_retire")
            if hasattr(self.params, 'current_age'):
                validate_age(self.params.current_age, "current_age")
            
            # Age consistency checks
            if hasattr(self.params, 'age_start') and hasattr(self.params, 'age_retire'):
                if self.params.age_start >= self.params.age_retire:
                    raise ValidationError(f"age_start ({self.params.age_start}) must be less than age_retire ({self.params.age_retire})")
            
            # Financial parameter validation
            if hasattr(self.params, 'annual_contribution'):
                validate_positive_value(self.params.annual_contribution, "annual_contribution")
                if self.params.annual_contribution > MAX_CONTRIBUTION:
                    raise ValidationError(f"annual_contribution ({self.params.annual_contribution:,.0f}) exceeds maximum ({MAX_CONTRIBUTION:,.0f})")
            
            if hasattr(self.params, 'initial_rurup'):
                validate_portfolio_value(self.params.initial_rurup, "initial_rurup")
            if hasattr(self.params, 'initial_broker'):
                validate_portfolio_value(self.params.initial_broker, "initial_broker")
            if hasattr(self.params, 'initial_l3'):
                validate_portfolio_value(self.params.initial_l3, "initial_l3")
            
            # Rate validation
            if hasattr(self.params, 'fund_fee'):
                validate_percentage(self.params.fund_fee, "fund_fee", max_percent=0.05)  # Max 5% fees
            if hasattr(self.params, 'pension_fee'):
                validate_percentage(self.params.pension_fee, "pension_fee", max_percent=0.05)
            if hasattr(self.params, 'cg_tax_normal'):
                validate_percentage(self.params.cg_tax_normal, "cg_tax_normal", max_percent=0.50)  # Max 50% tax
            if hasattr(self.params, 'tax_working'):
                validate_percentage(self.params.tax_working, "tax_working", max_percent=0.50)
            if hasattr(self.params, 'ruerup_tax'):
                validate_percentage(self.params.ruerup_tax, "ruerup_tax", max_percent=0.50)
            if hasattr(self.params, 'ruerup_ann_rate'):
                validate_percentage(self.params.ruerup_ann_rate, "ruerup_ann_rate", max_percent=0.20)  # Max 20% annuity rate
            if hasattr(self.params, 'discount_rate'):
                validate_positive_value(self.params.discount_rate, "discount_rate", allow_zero=False)
                if self.params.discount_rate > 0.15:  # Max 15% discount rate
                    raise ValidationError(f"discount_rate ({self.params.discount_rate:.1%}) exceeds reasonable maximum (15%)")
            
            # Spending validation
            if hasattr(self.params, 'desired_spend'):
                validate_positive_value(self.params.desired_spend, "desired_spend")
                if self.params.desired_spend > 1e6:  # Max 1M annual spending
                    raise ValidationError(f"desired_spend ({self.params.desired_spend:,.0f}) exceeds reasonable maximum (1,000,000)")
            if hasattr(self.params, 'public_pension'):
                validate_positive_value(self.params.public_pension, "public_pension")
            
            # Time period validation
            if hasattr(self.params, 'years_accum'):
                if not (1 <= self.params.years_accum <= 50):
                    raise ValidationError(f"years_accum ({self.params.years_accum}) must be between 1 and 50")
            if hasattr(self.params, 'years_transition'):
                if not (1 <= self.params.years_transition <= 20):
                    raise ValidationError(f"years_transition ({self.params.years_transition}) must be between 1 and 20")
            if hasattr(self.params, 'num_sims'):
                if not (100 <= self.params.num_sims <= 100000):
                    raise ValidationError(f"num_sims ({self.params.num_sims}) must be between 100 and 100,000")
            
            logger.info("Parameter validation completed successfully")
            
        except (ValidationError, NumericalStabilityError) as e:
            logger.error(f"Parameter validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during parameter validation: {e}")
            raise ValidationError(f"Parameter validation failed: {e}")

    def validate_pot(self, pot: Pot, context: str = "pot") -> Pot:
        """Validate pot values for numerical stability"""
        try:
            validate_portfolio_value(pot.rurup, f"{context}.rurup")
            validate_portfolio_value(pot.br_eq, f"{context}.br_eq")
            validate_portfolio_value(pot.br_eq_bs, f"{context}.br_eq_bs")
            validate_portfolio_value(pot.br_bd, f"{context}.br_bd")
            validate_portfolio_value(pot.br_bd_bs, f"{context}.br_bd_bs")
            validate_portfolio_value(pot.l3_eq, f"{context}.l3_eq")
            validate_portfolio_value(pot.l3_eq_bs, f"{context}.l3_eq_bs")
            
            # Consistency checks - Capital losses are expected in simulations, so demote to DEBUG
            if pot.br_eq_bs > pot.br_eq + 1e-6:
                logger.debug(f"{context}: br_eq_bs ({pot.br_eq_bs:,.0f}) > br_eq ({pot.br_eq:,.0f})")
            if pot.br_bd_bs > pot.br_bd + 1e-6:
                logger.debug(f"{context}: br_bd_bs ({pot.br_bd_bs:,.0f}) > br_bd ({pot.br_bd:,.0f})")
            if pot.l3_eq_bs > pot.l3_eq + 1e-6:
                logger.debug(f"{context}: l3_eq_bs ({pot.l3_eq_bs:,.0f}) > l3_eq ({pot.l3_eq:,.0f})")
            
            # Critical consistency checks - basis should never be negative
            if pot.br_eq_bs < -1e-6:
                logger.warning(f"{context}: negative br_eq_bs ({pot.br_eq_bs:,.0f})")
            if pot.br_bd_bs < -1e-6:
                logger.warning(f"{context}: negative br_bd_bs ({pot.br_bd_bs:,.0f})")
            if pot.l3_eq_bs < -1e-6:
                logger.warning(f"{context}: negative l3_eq_bs ({pot.l3_eq_bs:,.0f})")
            
            return pot
            
        except (ValidationError, NumericalStabilityError) as e:
            logger.error(f"Pot validation failed for {context}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during pot validation for {context}: {e}")
            raise NumericalStabilityError(f"Pot validation failed for {context}: {e}")

    def validate_returns(self, eq_returns: Union[float, List[float], None], bd_returns: Union[float, List[float], None] = None) -> Tuple[Union[float, List[float], None], Union[float, List[float], None]]:
        """Validate return values for numerical stability"""
        try:
            if eq_returns is not None:
                if isinstance(eq_returns, (list, tuple)):
                    for i, ret in enumerate(eq_returns):
                        validate_return_rate(ret, f"eq_returns[{i}]")
                else:
                    validate_return_rate(eq_returns, "eq_returns")
            
            if bd_returns is not None:
                if isinstance(bd_returns, (list, tuple)):
                    for i, ret in enumerate(bd_returns):
                        validate_return_rate(ret, f"bd_returns[{i}]")
                else:
                    validate_return_rate(bd_returns, "bd_returns")
            
            return eq_returns, bd_returns
            
        except (ValidationError, NumericalStabilityError) as e:
            logger.error(f"Returns validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during returns validation: {e}")
            raise NumericalStabilityError(f"Returns validation failed: {e}")

    def accumulate(self, eq_returns: np.ndarray | None = None, bd_returns: np.ndarray | None = None) -> Pot:
        raise NotImplementedError

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict[str, float],
    ) -> tuple[float, Pot]:
        raise NotImplementedError

    def transition_year(
        self, pot: Pot, current_year: int, eq_r: float, bd_r: float
    ) -> Pot:
        """Apply one year of transition with validation"""
        try:
            # Validate inputs
            pot = self.validate_pot(pot, f"transition_year_{current_year}_input")
            validate_return_rate(eq_r, f"eq_r_year_{current_year}")
            validate_return_rate(bd_r, f"bd_r_year_{current_year}")
            
            # Validate parameters exist
            if not hasattr(self.params, 'fund_fee'):
                raise ValidationError("fund_fee parameter missing")
            if not hasattr(self.params, 'pension_fee'):
                raise ValidationError("pension_fee parameter missing")
            
            # Calculate growth factors with validation
            rurup_growth = validate_calculation_result(
                (1 + eq_r) * (1 - self.params.fund_fee) * (1 - self.params.pension_fee),
                "rurup_growth_factor",
                {"eq_r": eq_r, "fund_fee": self.params.fund_fee, "pension_fee": self.params.pension_fee}
            )
            
            br_eq_growth = validate_calculation_result(
                (1 + eq_r) * (1 - self.params.fund_fee),
                "br_eq_growth_factor",
                {"eq_r": eq_r, "fund_fee": self.params.fund_fee}
            )
            
            l3_eq_growth = validate_calculation_result(
                (1 + eq_r) * (1 - self.params.fund_fee) * (1 - self.params.pension_fee),
                "l3_eq_growth_factor",
                {"eq_r": eq_r, "fund_fee": self.params.fund_fee, "pension_fee": self.params.pension_fee}
            )
            
            br_bd_growth = validate_calculation_result(
                (1 + bd_r) * (1 - self.params.fund_fee),
                "br_bd_growth_factor",
                {"bd_r": bd_r, "fund_fee": self.params.fund_fee}
            )
            
            # Apply growth with validation
            pot.rurup = safe_multiply(pot.rurup, rurup_growth)
            pot.br_eq = safe_multiply(pot.br_eq, br_eq_growth)
            pot.l3_eq = safe_multiply(pot.l3_eq, l3_eq_growth)
            pot.br_bd = safe_multiply(pot.br_bd, br_bd_growth)
            
            # Handle L3 to broker transition with validation
            if pot.l3_eq > 0:
                if not hasattr(self.params, 'years_transition'):
                    raise ValidationError("years_transition parameter missing")
                if not hasattr(self.params, 'ruerup_dist_fee'):
                    raise ValidationError("ruerup_dist_fee parameter missing")
                if not hasattr(self.params, 'cg_tax_normal'):
                    raise ValidationError("cg_tax_normal parameter missing")
                
                fraction = safe_divide(1.0, self.params.years_transition, "transition_fraction")
                moved = validate_calculation_result(
                    fraction * pot.l3_eq,
                    "l3_moved_amount",
                    {"fraction": fraction, "l3_eq": pot.l3_eq}
                )
                
                fees = validate_calculation_result(
                    moved * self.params.ruerup_dist_fee,
                    "l3_distribution_fees",
                    {"moved": moved, "ruerup_dist_fee": self.params.ruerup_dist_fee}
                )
                
                basis_moved = validate_calculation_result(
                    pot.l3_eq_bs * fraction,
                    "l3_basis_moved",
                    {"l3_eq_bs": pot.l3_eq_bs, "fraction": fraction}
                )
                
                gains = max(0, moved - basis_moved - fees)
                
                # Apply taxation with validation
                # Halbeinkünfteverfahren: Only 50% of gains are taxed at personal income tax rate
                # Also apply tfs_insurance (e.g. 15%)
                tfs_ins = getattr(self.params, 'tfs_insurance', 0.15)
                tax_rate_ins = getattr(self.params, 'tax_retirement', self.params.cg_tax_normal * 0.5)
                
                taxable_gains = gains * (1 - tfs_ins) * 0.5
                tax = validate_calculation_result(
                    taxable_gains * tax_rate_ins,
                    "l3_pension_tax",
                    {"taxable_gains": taxable_gains, "tax_rate": tax_rate_ins}
                )
                
                net_l3 = max(0, moved - tax - fees)
                
                # Update pot values with validation
                pot.br_bd = safe_add(pot.br_bd, net_l3)
                pot.br_bd_bs = safe_add(pot.br_bd_bs, basis_moved)
                pot.l3_eq = safe_subtract(pot.l3_eq, moved)
                pot.l3_eq_bs = safe_subtract(pot.l3_eq_bs, basis_moved)
            
            # Validate final pot state
            pot = self.validate_pot(pot, f"transition_year_{current_year}_output")
            
            return pot
            
        except (ValidationError, NumericalStabilityError) as e:
            logger.error(f"Validation error in transition_year {current_year}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in transition_year {current_year}: {e}")
            raise NumericalStabilityError(f"Transition year calculation failed: {e}")
