# ChakrView Step 59: Repository Task Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Task Specification (Step 59)
- **Scope**: Bounded Multi-File Python Repository Fixture for Dependency Reasoning & Repair

---

## 1. Objective & Challenge

To evaluate whether the ChakrView cognitive architecture can perform bounded repository-level problem solving, the task must satisfy specific structural properties:
1. **Multi-File Layout**: Multiple source modules and multiple test modules.
2. **Inter-Module Dependency Chain**: $A \longrightarrow B \longrightarrow C$.
3. **Decoupled Defect & Symptom**: A test fails on module $C$ (or the API layer), but the defect originates upstream in module $A$ (data representation) or module $B$ (business logic). Modifying only the symptomatic module $C$ is mathematically insufficient.
4. **Regression Protection**: Existing downstream modules and unit tests must continue to pass without regression.
5. **Diff Integrity**: Changes must be confined strictly to the files requiring modification.

---

## 2. Benchmark Fixture Layout (`OrderBillingRepository`)

The synthetic repository models an order processing and invoice calculation system:

```text
order_billing/
├── models.py         # Data structures: Item, Order, Invoice
├── tax_service.py    # Tax calculation: applies tax rates based on category
├── discount_engine.py# Discount logic: volume & VIP discount tiers
├── billing_service.py# Coordinates tax, discount, items to produce final Invoice
└── tests/
    ├── test_tax_service.py      # Unit tests for tax calculations
    ├── test_discount_engine.py  # Unit tests for discount tiers
    └── test_billing_service.py  # Integration tests for end-to-end invoice total
```

### Module Roles & Dependencies
- `models.py`: Declares `Item(name, price, category)`, `Order(order_id, items, customer_tier)`, `Invoice(subtotal, tax_amount, discount_amount, final_total)`.
- `tax_service.py`: Depends on `models.py`. Implements `compute_tax(order)`.
- `discount_engine.py`: Depends on `models.py`. Implements `compute_discount(order)`.
- `billing_service.py`: Depends on `models.py`, `tax_service.py`, `discount_engine.py`. Computes `generate_invoice(order)`.
- `tests/test_billing_service.py`: Integration test verifying `generate_invoice`.

---

## 3. The Controlled Defect Chain

### Upstream Defect (in `tax_service.py`):
In `tax_service.py`, tax for `"standard"` category items incorrectly computes tax as a flat offset instead of a rate multiplication:
```python
# DEFECTIVE CODE in tax_service.py:
def compute_tax(order: Order) -> float:
    total_tax = 0.0
    for item in order.items:
        if item.category == "standard":
            # BUG: adds flat 5.0 instead of 10% rate (0.10 * item.price)
            total_tax += 5.0
        elif item.category == "zero":
            total_tax += 0.0
    return round(total_tax, 2)
```

### Downstream Symptom (in `tests/test_billing_service.py`):
The integration test computes an invoice for an order with 2 items of price 50.0 (`subtotal = 100.0`):
- Expected tax: $10\% \times 100.0 = 10.0$.
- Expected final total: $100.0 - 0.0 + 10.0 = 110.0$.
- Actual test failure: `AssertionError: assert 120.0 == 110.0 (tax computed was 20.0 instead of 10.0 because two items each got +5.0 offset, but with another order configuration assert fails)`.

### Why Symptom Modification Fails:
Modifying `billing_service.py` to hardcode or subtract 10 breaks other valid orders and fails `test_tax_service.py`. The repair must identify that `billing_service.py` calls `tax_service.compute_tax`, trace the fault back to `tax_service.py`, correct the formula, and verify that both `test_tax_service.py` and `test_billing_service.py` pass.

---

## 4. Verification Requirements (4 Levels)

1. **Level 1 — Targeted Verification**: `test_billing_service.py` passes cleanly.
2. **Level 2 — Regression Verification**: `test_tax_service.py` and `test_discount_engine.py` pass cleanly.
3. **Level 3 — Repository State Verification**: All 3 test suites pass simultaneously ($100\%$ pass rate).
4. **Level 4 — Diff Integrity**: Only `tax_service.py` has diff modifications; other source files and test suites remain untouched.
