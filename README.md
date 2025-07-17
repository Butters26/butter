# Invoice Payment System

A simple Python-based invoice payment system that allows you to create, manage, and pay invoices.

## Features

- ✅ Create and manage invoices
- 💳 Process payments with multiple payment methods
- 📊 Track payment history
- 💾 Persistent data storage (JSON)
- 🖥️ Command-line interface

## Quick Start

### Pay an Invoice Immediately

```bash
# Create and pay a sample invoice
python3 create_and_pay.py
```

### Use Interactive CLI

```bash
# Launch interactive invoice payment system
python3 pay_invoice_cli.py
```

### Quick Payment

```bash
# Pay the first pending invoice quickly
python3 quick_pay.py
```

## Files

- `invoice_payment.py` - Core invoice payment system
- `pay_invoice_cli.py` - Interactive command-line interface
- `quick_pay.py` - Quick payment script
- `create_and_pay.py` - Demo script that creates and pays an invoice
- `invoices_data.json` - Data storage file (auto-generated)

## Payment Methods

- Credit Card
- Bank Transfer
- PayPal
- Cash

## Example Usage

```python
from invoice_payment import InvoicePaymentSystem, PaymentMethod

# Create system
system = InvoicePaymentSystem()

# Create invoice
invoice = system.create_invoice(
    invoice_id="INV-001",
    amount=100.0,
    currency="USD",
    description="Service Fee",
    due_date="2024-12-31"
)

# Pay invoice
result = system.pay_invoice("INV-001", PaymentMethod.CREDIT_CARD)
print(f"Payment status: {result['status']}")
```