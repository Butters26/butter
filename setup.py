#!/usr/bin/env python3
"""
Setup script for the Bug Detection Tools
Installs required dependencies and sets up the environment.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")


def install_python_dependencies():
    """Install Python dependencies"""
    dependencies = [
        "pyyaml",  # For YAML config files
        "pylint",  # Python linter
        "flake8",  # Python style checker
        "bandit",  # Security analysis
        "mypy",    # Type checking
        "black",   # Code formatter
        "isort",   # Import sorter
    ]
    
    print("📦 Installing Python dependencies...")
    
    for dep in dependencies:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing anyway...")


def install_node_dependencies():
    """Install Node.js dependencies for JavaScript analysis"""
    if not shutil.which("npm"):
        print("⚠️  npm not found. Install Node.js for JavaScript analysis.")
        return
    
    print("📦 Installing Node.js dependencies...")
    
    js_tools = ["eslint", "jshint", "prettier"]
    
    for tool in js_tools:
        success = run_command(f"npm install -g {tool}", f"Installing {tool}")
        if not success:
            print(f"⚠️  Failed to install {tool}, continuing anyway...")


def install_system_tools():
    """Install system-level tools"""
    print("🔧 Checking system tools...")
    
    # Check for shellcheck
    if not shutil.which("shellcheck"):
        print("⚠️  shellcheck not found. Install it for shell script analysis:")
        print("   Ubuntu/Debian: sudo apt-get install shellcheck")
        print("   macOS: brew install shellcheck")
        print("   Other: https://github.com/koalaman/shellcheck#installing")
    else:
        print("✅ shellcheck found")
    
    # Check for other useful tools
    tools = {
        "git": "Version control (usually pre-installed)",
        "yamllint": "YAML file linter (pip install yamllint)",
        "hadolint": "Dockerfile linter (see https://github.com/hadolint/hadolint)"
    }
    
    for tool, description in tools.items():
        if shutil.which(tool):
            print(f"✅ {tool} found")
        else:
            print(f"⚠️  {tool} not found - {description}")


def make_executable():
    """Make the scripts executable"""
    scripts = ["bug_detector.py", "enhanced_bug_detector.py"]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"✅ Made {script} executable")


def create_symlinks():
    """Create convenient symlinks"""
    scripts = {
        "bug_detector.py": "bugdet",
        "enhanced_bug_detector.py": "ebugdet"
    }
    
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    
    for script, symlink in scripts.items():
        if os.path.exists(script):
            script_path = Path.cwd() / script
            symlink_path = local_bin / symlink
            
            try:
                if symlink_path.exists():
                    symlink_path.unlink()
                symlink_path.symlink_to(script_path)
                print(f"✅ Created symlink: {symlink} -> {script}")
            except Exception as e:
                print(f"⚠️  Failed to create symlink {symlink}: {e}")
    
    print(f"\n💡 Add {local_bin} to your PATH to use 'bugdet' and 'ebugdet' commands")
    print(f"   Add this to your ~/.bashrc or ~/.zshrc:")
    print(f"   export PATH=\"{local_bin}:$PATH\"")


def main():
    print("🚀 Setting up Bug Detection Tools")
    print("=" * 50)
    
    check_python_version()
    install_python_dependencies()
    install_node_dependencies()
    install_system_tools()
    make_executable()
    create_symlinks()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nQuick start:")
    print("  python bug_detector.py --help")
    print("  python enhanced_bug_detector.py --help")
    print("\nOr if you added ~/.local/bin to PATH:")
    print("  bugdet --help")
    print("  ebugdet --help")
    
    print("\nExample usage:")
    print("  python bug_detector.py .")
    print("  python enhanced_bug_detector.py . -r -f html -o report.html")


if __name__ == "__main__":
    main()