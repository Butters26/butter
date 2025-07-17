#!/usr/bin/env python3
"""
Example Python file with intentional bugs and issues for testing
the bug detection tools.
"""

import os
import sys

# Security issue: hardcoded password
DATABASE_PASSWORD = "supersecret123456"
API_KEY = 'abc123-very-secret-key-456'

# TODO: Fix this security vulnerability
def authenticate_user(username, password):
    # Bad practice: using eval (security risk)
    result = eval(f"'{username}' == 'admin'")
    
    # Undefined variable error
    if result and password == DATABASE_PASSWORD:
        print("Authentication successful!")  # Debug statement
        return authenticated_status  # NameError: undefined variable
    
    return False

# Function with high complexity
def complex_function(data, mode, options, debug=False):
    if data is None:
        if mode == "strict":
            if options.get("throw_error"):
                if debug:
                    if len(options) > 5:
                        if "advanced" in options:
                            if options["advanced"]["level"] > 10:
                                raise ValueError("Too complex")
                            else:
                                return None
                        else:
                            return {}
                    else:
                        return []
                else:
                    return False
            else:
                return True
        else:
            return None
    return data

# Bad practices in exception handling
def risky_operation():
    try:
        # Potential SQL injection
        query = "SELECT * FROM users WHERE name = '%s'" % user_input
        execute(query)
    except:  # Bare except clause
        pass  # Silent failure

# Comparison issues
def check_values(x, y):
    # Using 'is' for value comparison instead of '=='
    if x is 5:
        return True
    
    # This could be using == instead of ===  (not applicable in Python but good to test)
    return x == y

# Line that is way too long to demonstrate line length checking - this line exceeds 120 characters and should be flagged
very_long_variable_name_that_makes_this_line_exceed_reasonable_length = "This is a very long string that contributes to making this line way too long"

def main():
    username = input("Username: ")
    password = input("Password: ")
    
    # More debug statements
    print(f"Attempting login for {username}")
    print(f"Password length: {len(password)}")
    
    if authenticate_user(username, password):
        print("Welcome!")
    else:
        print("Access denied")

# Missing if __name__ == "__main__" guard
main()

# FIXME: This whole file needs refactoring
# HACK: Temporary workaround
# BUG: Known issue with authentication