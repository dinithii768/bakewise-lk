"""
Calculator Tool for BakeWise LK
Provides pricing and EPF/ETF calculations
for Sri Lankan home food businesses
"""


def calculate_product_price(
    ingredient_cost: float,
    packaging_cost: float,
    overhead_per_unit: float,
    labor_per_unit: float,
    markup_percent: float = 40.0
) -> dict:
    """Calculate recommended selling price for a food product."""
    total_cost = (
        ingredient_cost +
        packaging_cost +
        overhead_per_unit +
        labor_per_unit
    )
    profit = total_cost * (markup_percent / 100)
    selling_price = total_cost + profit

    return {
        "ingredient_cost_lkr": round(ingredient_cost, 2),
        "packaging_cost_lkr": round(packaging_cost, 2),
        "overhead_per_unit_lkr": round(overhead_per_unit, 2),
        "labor_per_unit_lkr": round(labor_per_unit, 2),
        "total_cost_lkr": round(total_cost, 2),
        "markup_percent": markup_percent,
        "profit_per_unit_lkr": round(profit, 2),
        "recommended_price_lkr": round(selling_price, 2)
    }


def calculate_batch_profit(
    selling_price: float,
    total_cost_per_unit: float,
    units_per_batch: int,
    batches_per_month: int
) -> dict:
    """Calculate monthly profit from selling baked goods."""
    profit_per_unit = selling_price - total_cost_per_unit
    profit_per_batch = profit_per_unit * units_per_batch
    monthly_revenue = selling_price * units_per_batch * batches_per_month
    monthly_cost = total_cost_per_unit * units_per_batch * batches_per_month
    monthly_profit = monthly_revenue - monthly_cost

    return {
        "profit_per_unit_lkr": round(profit_per_unit, 2),
        "profit_per_batch_lkr": round(profit_per_batch, 2),
        "monthly_revenue_lkr": round(monthly_revenue, 2),
        "monthly_cost_lkr": round(monthly_cost, 2),
        "monthly_profit_lkr": round(monthly_profit, 2),
        "units_per_month": units_per_batch * batches_per_month
    }


def calculate_epf_etf(
    monthly_salary: float,
    num_employees: int = 1
) -> dict:
    """
    Calculate EPF and ETF for Sri Lankan food business staff.
    EPF: Employee 8%, Employer 12%
    ETF: Employer 3%
    """
    employee_epf = monthly_salary * 0.08
    employer_epf = monthly_salary * 0.12
    employer_etf = monthly_salary * 0.03
    total_employer_cost = monthly_salary + employer_epf + employer_etf
    employee_take_home = monthly_salary - employee_epf

    per_employee = {
        "gross_salary_lkr": round(monthly_salary, 2),
        "employee_epf_deduction_lkr": round(employee_epf, 2),
        "employee_take_home_lkr": round(employee_take_home, 2),
        "employer_epf_lkr": round(employer_epf, 2),
        "employer_etf_lkr": round(employer_etf, 2),
        "total_employer_cost_lkr": round(total_employer_cost, 2)
    }

    total = {
        "total_epf_lkr": round((employee_epf + employer_epf) * num_employees, 2),
        "total_etf_lkr": round(employer_etf * num_employees, 2),
        "total_employer_burden_lkr": round(total_employer_cost * num_employees, 2)
    }

    return {
        "per_employee": per_employee,
        "total_for_all_employees": total,
        "num_employees": num_employees
    }


def calculate_vat_status(annual_revenue: float) -> dict:
    """Check VAT registration requirement for Sri Lankan businesses."""
    threshold = 60_000_000
    quarterly_threshold = 15_000_000
    quarterly_revenue = annual_revenue / 4

    must_register = annual_revenue >= threshold

    return {
        "annual_revenue_lkr": annual_revenue,
        "quarterly_revenue_lkr": round(quarterly_revenue, 2),
        "vat_annual_threshold_lkr": threshold,
        "vat_quarterly_threshold_lkr": quarterly_threshold,
        "must_register_for_vat": must_register,
        "vat_rate_percent": 18.0,
        "advice": (
            "You must register for VAT with the IRD."
            if must_register
            else f"You are below the VAT threshold. "
                 f"LKR {(threshold - annual_revenue):,.0f} "
                 f"away from mandatory registration."
        )
    }