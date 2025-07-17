#!/usr/bin/env python3
"""
Invoice Payment System
A simple system to handle invoice payments
"""

import json
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class PaymentStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    CASH = "cash"

@dataclass
class Invoice:
    invoice_id: str
    amount: float
    currency: str
    description: str
    due_date: str
    status: PaymentStatus = PaymentStatus.PENDING
    created_date: str = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.datetime.now().isoformat()

@dataclass
class Payment:
    payment_id: str
    invoice_id: str
    amount: float
    payment_method: PaymentMethod
    payment_date: str
    status: PaymentStatus = PaymentStatus.PENDING
    
    def __post_init__(self):
        if self.payment_date is None:
            self.payment_date = datetime.datetime.now().isoformat()

class InvoicePaymentSystem:
    def __init__(self):
        self.invoices: Dict[str, Invoice] = {}
        self.payments: Dict[str, Payment] = {}
        
    def create_invoice(self, invoice_id: str, amount: float, currency: str, 
                      description: str, due_date: str) -> Invoice:
        """Create a new invoice"""
        invoice = Invoice(
            invoice_id=invoice_id,
            amount=amount,
            currency=currency,
            description=description,
            due_date=due_date
        )
        self.invoices[invoice_id] = invoice
        return invoice
    
    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get an invoice by ID"""
        return self.invoices.get(invoice_id)
    
    def list_pending_invoices(self) -> List[Invoice]:
        """List all pending invoices"""
        return [invoice for invoice in self.invoices.values() 
                if invoice.status == PaymentStatus.PENDING]
    
    def pay_invoice(self, invoice_id: str, payment_method: PaymentMethod, 
                   payment_id: str = None) -> Dict[str, str]:
        """Pay an invoice"""
        if payment_id is None:
            payment_id = f"pay_{invoice_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        invoice = self.get_invoice(invoice_id)
        if not invoice:
            return {"status": "error", "message": f"Invoice {invoice_id} not found"}
        
        if invoice.status == PaymentStatus.PAID:
            return {"status": "error", "message": f"Invoice {invoice_id} is already paid"}
        
        # Create payment record
        payment = Payment(
            payment_id=payment_id,
            invoice_id=invoice_id,
            amount=invoice.amount,
            payment_method=payment_method,
            payment_date=datetime.datetime.now().isoformat(),
            status=PaymentStatus.PAID
        )
        
        # Update invoice status
        invoice.status = PaymentStatus.PAID
        self.payments[payment_id] = payment
        
        return {
            "status": "success",
            "message": f"Invoice {invoice_id} paid successfully",
            "payment_id": payment_id,
            "amount": invoice.amount,
            "currency": invoice.currency
        }
    
    def get_payment_history(self, invoice_id: str = None) -> List[Payment]:
        """Get payment history for a specific invoice or all payments"""
        if invoice_id:
            return [payment for payment in self.payments.values() 
                   if payment.invoice_id == invoice_id]
        return list(self.payments.values())
    
    def save_to_file(self, filename: str = "invoices_data.json"):
        """Save invoices and payments to file"""
        data = {
            "invoices": {},
            "payments": {}
        }
        
        # Convert invoices to dict with enum values
        for k, v in self.invoices.items():
            invoice_dict = asdict(v)
            invoice_dict["status"] = v.status.value
            data["invoices"][k] = invoice_dict
        
        # Convert payments to dict with enum values
        for k, v in self.payments.items():
            payment_dict = asdict(v)
            payment_dict["status"] = v.status.value
            payment_dict["payment_method"] = v.payment_method.value
            data["payments"][k] = payment_dict
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, filename: str = "invoices_data.json"):
        """Load invoices and payments from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for k, v in data.get("invoices", {}).items():
                v["status"] = PaymentStatus(v["status"])
                self.invoices[k] = Invoice(**v)
            
            for k, v in data.get("payments", {}).items():
                v["status"] = PaymentStatus(v["status"])
                v["payment_method"] = PaymentMethod(v["payment_method"])
                self.payments[k] = Payment(**v)
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with empty system.")

def main():
    """Main function to demonstrate invoice payment"""
    system = InvoicePaymentSystem()
    
    # Try to load existing data
    system.load_from_file()
    
    print("=== Invoice Payment System ===")
    print()
    
    # List pending invoices
    pending_invoices = system.list_pending_invoices()
    
    if not pending_invoices:
        print("No pending invoices found. Creating a sample invoice...")
        # Create a sample invoice
        sample_invoice = system.create_invoice(
            invoice_id="INV-001",
            amount=1250.00,
            currency="USD",
            description="Web Development Services",
            due_date="2024-12-31"
        )
        pending_invoices = [sample_invoice]
        print(f"Created sample invoice: {sample_invoice.invoice_id}")
        print()
    
    print("Pending Invoices:")
    for invoice in pending_invoices:
        print(f"  ID: {invoice.invoice_id}")
        print(f"  Amount: {invoice.amount} {invoice.currency}")
        print(f"  Description: {invoice.description}")
        print(f"  Due Date: {invoice.due_date}")
        print(f"  Status: {invoice.status.value}")
        print()
    
    # Pay the first pending invoice
    if pending_invoices:
        invoice_to_pay = pending_invoices[0]
        print(f"Paying invoice {invoice_to_pay.invoice_id}...")
        
        result = system.pay_invoice(
            invoice_id=invoice_to_pay.invoice_id,
            payment_method=PaymentMethod.CREDIT_CARD
        )
        
        if result["status"] == "success":
            print("✅ Payment successful!")
            print(f"   Payment ID: {result['payment_id']}")
            print(f"   Amount: {result['amount']} {result['currency']}")
        else:
            print(f"❌ Payment failed: {result['message']}")
        
        print()
        
        # Show updated invoice status
        updated_invoice = system.get_invoice(invoice_to_pay.invoice_id)
        print(f"Updated invoice status: {updated_invoice.status.value}")
        
        # Show payment history
        payment_history = system.get_payment_history(invoice_to_pay.invoice_id)
        if payment_history:
            print("\nPayment History:")
            for payment in payment_history:
                print(f"  Payment ID: {payment.payment_id}")
                print(f"  Amount: {payment.amount}")
                print(f"  Method: {payment.payment_method.value}")
                print(f"  Date: {payment.payment_date}")
                print(f"  Status: {payment.status.value}")
    
    # Save data
    system.save_to_file()
    print("\nData saved to invoices_data.json")

if __name__ == "__main__":
    main()