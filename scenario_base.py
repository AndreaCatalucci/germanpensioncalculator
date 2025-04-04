from dataclasses import dataclass


@dataclass
class Pot:
    rurup: float = 0.0

    br_eq: float = 0.0
    br_eq_bs: float = 0.0
    br_bd: float = 0.0
    br_bd_bs: float = 0.0

    l3_eq: float = 0.0
    l3_eq_bs: float = 0.0

    def leftover(self):
        return self.br_eq + self.br_bd + self.l3_eq


@dataclass
class Withdrawal:
    gross_withdrawn: float
    net_withdrawn: float


def present_value(cf, year, discount_rate):
    return cf / ((1 + discount_rate) ** year)


def withdraw(pot: Pot, net_needed, cg_tax) -> Withdrawal:
    total_pot = pot.br_bd + pot.br_eq + pot.l3_eq
    if total_pot <= 0:
        return Withdrawal(0.0, 0.0)

    net_withdrawn = 0
    gross_withdrawn = 0
    still_needed = net_needed

    # BOND FIRST
    if pot.br_bd > 0:
        net_bd = pot.br_bd - (pot.br_bd - pot.br_bd_bs) * cg_tax
        portion = min(still_needed, net_bd)
        if portion > 0:
            frac = portion / net_bd
            gross_bd_withdrawal = frac * pot.br_bd
            pot.br_bd_bs = pot.br_bd_bs * (1 - frac)
            pot.br_bd -= gross_bd_withdrawal
            net_withdrawn += portion
            gross_withdrawn += gross_bd_withdrawal
            still_needed -= portion

    # EQUITY NEXT
    if still_needed > 0 and pot.br_eq > 0:
        net_eq = pot.br_eq - (pot.br_eq - pot.br_eq_bs) * cg_tax
        portion2 = min(still_needed, net_eq)
        if portion2 > 0:
            frac2 = portion2 / net_eq
            gross_eq_withdrawal = frac2 * pot.br_eq
            pot.br_eq_bs = pot.br_eq_bs * (1 - frac2)
            pot.br_eq -= gross_eq_withdrawal
            net_withdrawn += portion2
            gross_withdrawn += gross_eq_withdrawal
            still_needed -= portion2

    return Withdrawal(gross_withdrawn, net_withdrawn)


def shift_equity_to_bonds(pot: Pot, fraction, cgt):
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
    def __init__(self, p):
        self.params = p

    def accumulate(self, eq_returns=None, bd_returns=None) -> Pot:
        raise NotImplementedError

    def decumulate_year(
        self,
        pot: Pot,
        current_year: int,
        net_ann: float,
        needed_net: float,
        rand_returns: dict,
    ) -> tuple[float, Pot]:
        raise NotImplementedError

    def transition_year(
        self, pot: Pot, current_year: int, eq_r: float, bd_r: float
    ) -> Pot:
        pot.rurup *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
        pot.br_eq *= 1 + eq_r - self.params.fund_fee
        pot.l3_eq *= 1 + eq_r - self.params.fund_fee - self.params.pension_fee
        pot.br_bd *= 1 + bd_r - self.params.fund_fee
        if pot.l3_eq > 0:
            fraction = 1.0 / self.params.years_transition
            moved = fraction * pot.l3_eq
            fees = moved * self.params.ruerup_dist_fee
            gains = max(0, moved - pot.l3_eq_bs * fraction - fees)
            tax = gains * self.params.cg_tax_half
            net_l3 = max(0, moved - tax - fees)
            pot.br_bd += net_l3
            pot.br_bd_bs += net_l3
            pot.l3_eq -= moved
            pot.l3_eq_bs -= pot.l3_eq_bs * fraction
