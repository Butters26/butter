#!/usr/bin/env python3
"""
Create and Pay Invoice Script
Creates a sample invoice and pays it immediately
"""

from invoice_payment import InvoicePaymentSystem, PaymentMethod

def create_and_pay():
    """Create a sample invoice and pay it"""
    system = InvoicePaymentSystem()
    system.load_from_file()
    
    print("🎯 Creating and paying an invoice...")
    
    # Create a sample invoice
    invoice = system.create_invoice(
        invoice_id="INV-2024-001",
        amount=500.00,
        currency="USD",
        description="Consulting Services - January 2024",
        due_date="2024-12-31"
    )
    
    print(f"\n📄 Created invoice:")
    print(f"   ID: {invoice.invoice_id}")
    print(f"   Amount: {invoice.amount} {invoice.currency}")
    print(f"   Description: {invoice.description}")
    print(f"   Due Date: {invoice.due_date}")
    print(f"   Status: {invoice.status.value}")
    
    # Pay the invoice
    print(f"\n💳 Processing payment...")
    result = system.pay_invoice(
        invoice_id=invoice.invoice_id,
        payment_method=PaymentMethod.CREDIT_CARD
    )
    
    if result["status"] == "success":
        print("\n🎉 Payment completed successfully!")
        print(f"   Payment ID: {result['payment_id']}")
        print(f"   Amount paid: {result['amount']} {result['currency']}")
        print(f"   Payment method: Credit Card")
        
        # Show updated invoice status
        updated_invoice = system.get_invoice(invoice.invoice_id)
        print(f"   Invoice status: {updated_invoice.status.value}")
    else:
        print(f"\n❌ Payment failed: {result['message']}")
    
    # Save data
    system.save_to_file()
    print("\n💾 Data saved successfully!")

if __name__ == "__main__":
    create_and_pay()