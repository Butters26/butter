#!/usr/bin/env python3
"""
Interactive Invoice Payment CLI
Simple command-line interface for paying invoices
"""

import sys
from invoice_payment import InvoicePaymentSystem, PaymentMethod, PaymentStatus

def display_menu():
    """Display the main menu"""
    print("\n=== Invoice Payment CLI ===")
    print("1. List pending invoices")
    print("2. Pay an invoice")
    print("3. View payment history")
    print("4. Create new invoice")
    print("5. Exit")
    return input("\nSelect an option (1-5): ").strip()

def list_pending_invoices(system):
    """List all pending invoices"""
    pending_invoices = system.list_pending_invoices()
    
    if not pending_invoices:
        print("\n📋 No pending invoices found.")
        return
    
    print(f"\n📋 Found {len(pending_invoices)} pending invoice(s):")
    print("-" * 60)
    
    for invoice in pending_invoices:
        print(f"Invoice ID: {invoice.invoice_id}")
        print(f"Amount: {invoice.amount} {invoice.currency}")
        print(f"Description: {invoice.description}")
        print(f"Due Date: {invoice.due_date}")
        print(f"Status: {invoice.status.value}")
        print("-" * 60)

def pay_invoice_interactive(system):
    """Interactive invoice payment"""
    pending_invoices = system.list_pending_invoices()
    
    if not pending_invoices:
        print("\n❌ No pending invoices to pay.")
        return
    
    # Show pending invoices
    list_pending_invoices(system)
    
    # Get invoice ID to pay
    invoice_id = input("\nEnter the Invoice ID to pay: ").strip()
    
    invoice = system.get_invoice(invoice_id)
    if not invoice:
        print(f"❌ Invoice {invoice_id} not found.")
        return
    
    if invoice.status == PaymentStatus.PAID:
        print(f"❌ Invoice {invoice_id} is already paid.")
        return
    
    # Show invoice details
    print(f"\n💰 Invoice Details:")
    print(f"   ID: {invoice.invoice_id}")
    print(f"   Amount: {invoice.amount} {invoice.currency}")
    print(f"   Description: {invoice.description}")
    print(f"   Due Date: {invoice.due_date}")
    
    # Confirm payment
    confirm = input(f"\nConfirm payment of {invoice.amount} {invoice.currency}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Payment cancelled.")
        return
    
    # Select payment method
    print("\nSelect payment method:")
    methods = list(PaymentMethod)
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method.value.replace('_', ' ').title()}")
    
    try:
        method_choice = int(input("\nEnter payment method (1-4): ").strip())
        if 1 <= method_choice <= len(methods):
            payment_method = methods[method_choice - 1]
        else:
            print("❌ Invalid payment method selection.")
            return
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return
    
    # Process payment
    result = system.pay_invoice(invoice_id, payment_method)
    
    if result["status"] == "success":
        print("\n✅ Payment successful!")
        print(f"   Payment ID: {result['payment_id']}")
        print(f"   Amount: {result['amount']} {result['currency']}")
        print(f"   Method: {payment_method.value.replace('_', ' ').title()}")
    else:
        print(f"\n❌ Payment failed: {result['message']}")

def view_payment_history(system):
    """View payment history"""
    payments = system.get_payment_history()
    
    if not payments:
        print("\n📜 No payment history found.")
        return
    
    print(f"\n📜 Payment History ({len(payments)} payment(s)):")
    print("-" * 80)
    
    for payment in payments:
        print(f"Payment ID: {payment.payment_id}")
        print(f"Invoice ID: {payment.invoice_id}")
        print(f"Amount: {payment.amount}")
        print(f"Method: {payment.payment_method.value.replace('_', ' ').title()}")
        print(f"Date: {payment.payment_date}")
        print(f"Status: {payment.status.value}")
        print("-" * 80)

def create_invoice_interactive(system):
    """Create a new invoice interactively"""
    print("\n📝 Create New Invoice")
    
    try:
        invoice_id = input("Invoice ID: ").strip()
        if not invoice_id:
            print("❌ Invoice ID cannot be empty.")
            return
        
        if system.get_invoice(invoice_id):
            print(f"❌ Invoice {invoice_id} already exists.")
            return
        
        amount = float(input("Amount: ").strip())
        currency = input("Currency (e.g., USD, EUR): ").strip().upper()
        description = input("Description: ").strip()
        due_date = input("Due Date (YYYY-MM-DD): ").strip()
        
        invoice = system.create_invoice(invoice_id, amount, currency, description, due_date)
        
        print("\n✅ Invoice created successfully!")
        print(f"   ID: {invoice.invoice_id}")
        print(f"   Amount: {invoice.amount} {invoice.currency}")
        print(f"   Description: {invoice.description}")
        print(f"   Due Date: {invoice.due_date}")
        
    except ValueError:
        print("❌ Invalid amount. Please enter a valid number.")
    except Exception as e:
        print(f"❌ Error creating invoice: {str(e)}")

def main():
    """Main CLI loop"""
    system = InvoicePaymentSystem()
    system.load_from_file()
    
    print("🎯 Welcome to the Invoice Payment System!")
    
    while True:
        try:
            choice = display_menu()
            
            if choice == "1":
                list_pending_invoices(system)
            elif choice == "2":
                pay_invoice_interactive(system)
            elif choice == "3":
                view_payment_history(system)
            elif choice == "4":
                create_invoice_interactive(system)
            elif choice == "5":
                print("\n👋 Thank you for using the Invoice Payment System!")
                break
            else:
                print("\n❌ Invalid option. Please select 1-5.")
            
            # Save data after each operation
            system.save_to_file()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()