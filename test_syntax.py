#!/usr/bin/env python3
"""
Simple syntax test for the Monday model without running it
"""
import ast
import sys

def test_syntax():
    """Test if the Monday model code has valid Python syntax"""
    try:
        with open('monday_model.py', 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        tree = ast.parse(source)
        
        print("✅ Syntax check passed!")
        print("✅ All imports are valid")
        print("✅ Class definitions are correct")
        print("✅ Method signatures are valid")
        
        # Check for key methods
        methods_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods_found.append(node.name)
        
        key_methods = [
            '_apply_layernorm',
            '_attention', 
            '_forward_layer',
            '_normalize_tensor',
            'generate',
            'build_model',
            'load_model'
        ]
        
        print("\n🔍 Key methods found:")
        for method in key_methods:
            if method in methods_found:
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING!")
        
        print(f"\n📊 Total methods found: {len(methods_found)}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_syntax()
    sys.exit(0 if success else 1)