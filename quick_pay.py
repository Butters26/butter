#!/usr/bin/env python3
"""
Quick Invoice Payment Script
Quickly pay the first pending invoice
"""

from invoice_payment import InvoicePaymentSystem, PaymentMethod

def quick_pay():
    """Pay the first pending invoice quickly"""
    system = InvoicePaymentSystem()
    system.load_from_file()
    
    # Get pending invoices
    pending_invoices = system.list_pending_invoices()
    
    if not pending_invoices:
        print("✅ No pending invoices to pay!")
        return
    
    # Pay the first pending invoice
    invoice = pending_invoices[0]
    print(f"💳 Paying invoice {invoice.invoice_id}...")
    print(f"   Amount: {invoice.amount} {invoice.currency}")
    print(f"   Description: {invoice.description}")
    
    result = system.pay_invoice(
        invoice_id=invoice.invoice_id,
        payment_method=PaymentMethod.CREDIT_CARD  # Default to credit card
    )
    
    if result["status"] == "success":
        print("\n🎉 Payment successful!")
        print(f"   Payment ID: {result['payment_id']}")
        print(f"   Amount paid: {result['amount']} {result['currency']}")
        
        # Check for more pending invoices
        remaining = system.list_pending_invoices()
        if remaining:
            print(f"\n📋 {len(remaining)} invoice(s) still pending.")
        else:
            print("\n✅ All invoices are now paid!")
    else:
        print(f"\n❌ Payment failed: {result['message']}")
    
    # Save data
    system.save_to_file()

if __name__ == "__main__":
    quick_pay()