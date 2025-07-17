# 🐛 Comprehensive Bug Detection Tools

A powerful suite of static analysis tools that can find and detect errors, bugs, and code quality issues in your codebase across multiple programming languages.

## 🌟 Features

### Core Capabilities
- **Multi-language support**: Python, JavaScript/TypeScript, Shell scripts, and general text analysis
- **AST-based analysis**: Deep code understanding for Python files
- **Security vulnerability detection**: Find hardcoded secrets, SQL injection risks, XSS vulnerabilities
- **Code quality checks**: Style issues, complexity analysis, best practice violations
- **External tool integration**: Seamlessly integrates with popular linters (pylint, flake8, ESLint, etc.)
- **Configurable rules**: Customize analysis through configuration files
- **Multiple output formats**: Text, JSON, and HTML reports
- **Parallel processing**: Fast analysis of large codebases

### Two Versions Available

1. **`bug_detector.py`** - Core bug detection with built-in analyzers
2. **`enhanced_bug_detector.py`** - Advanced version with external tool integration and configuration

## 🚀 Quick Start

### Installation

1. Clone or download the tools:
```bash
# The tools are ready to use!
python setup.py  # Run setup for dependencies and configuration
```

2. Make scripts executable:
```bash
chmod +x bug_detector.py enhanced_bug_detector.py
```

### Basic Usage

```bash
# Analyze current directory
python bug_detector.py .

# Analyze specific file
python bug_detector.py myfile.py

# Recursive analysis with enhanced tool
python enhanced_bug_detector.py . -r

# Generate HTML report
python enhanced_bug_detector.py . -r -f html -o report.html

# Show only high severity issues
python enhanced_bug_detector.py . --severity HIGH
```

## 📋 Command Line Options

### Basic Bug Detector (`bug_detector.py`)
```
usage: bug_detector.py [-h] [-r] [-f {text,json}] [-o OUTPUT] [--severity {LOW,MEDIUM,HIGH,CRITICAL}] path

positional arguments:
  path                  File or directory to analyze

optional arguments:
  -r, --recursive       Recursively analyze directories
  -f, --format         Output format (text, json)
  -o, --output         Output file (default: stdout)
  --severity           Minimum severity level to report
```

### Enhanced Bug Detector (`enhanced_bug_detector.py`)
```
usage: enhanced_bug_detector.py [-h] [-r] [-f {text,json,html}] [-o OUTPUT] [-c CONFIG] [--create-config CREATE_CONFIG] [--severity {LOW,MEDIUM,HIGH,CRITICAL}] [--tools] path

positional arguments:
  path                  File or directory to analyze

optional arguments:
  -r, --recursive       Recursively analyze directories
  -f, --format         Output format (text, json, html)
  -o, --output         Output file (default: stdout)
  -c, --config         Configuration file path
  --create-config      Create default configuration file
  --severity           Minimum severity level to report
  --tools              Show available external tools
```

## ⚙️ Configuration

The enhanced bug detector supports configuration files in JSON or YAML format:

```bash
# Create default configuration
python enhanced_bug_detector.py --create-config config.yaml

# Use custom configuration
python enhanced_bug_detector.py . -c config.yaml
```

### Example Configuration (YAML)
```yaml
rules:
  max_line_length: 120
  check_debug_statements: true
  check_todos: true
  check_security_issues: true
  check_code_style: true
  check_complexity: true
  max_complexity: 10

severity_overrides:
  E501: LOW  # Line too long - flake8
  W503: LOW  # Line break before binary operator

ignore_patterns:
  - "*.pyc"
  - "__pycache__/*"
  - "node_modules/*"
  - ".git/*"
  - "*.min.js"

external_tools:
  enabled: true
  python:
    - pylint
    - flake8
    - bandit
  javascript:
    - eslint
  shell:
    - shellcheck
```

## 🔍 What the Tools Detect

### Python-Specific Issues
- **Syntax errors** and **undefined variables**
- **Security risks**: `eval()`, `exec()`, SQL injection patterns
- **Bad practices**: Bare except clauses, inappropriate use of `is` operator
- **Code complexity**: Functions with high cyclomatic complexity
- **Style issues**: Long lines, debug print statements, TODO comments

### JavaScript/TypeScript Issues
- **Equality checks**: Use of `==` instead of `===`
- **Legacy syntax**: `var` instead of `let`/`const`
- **Security risks**: `eval()` usage, potential XSS with `innerHTML`
- **Debug statements**: `console.log` calls

### Shell Script Issues
- **Unquoted variables** that could cause word splitting
- **Missing error handling**: Suggestions for `set -e`
- **Shellcheck integration** for comprehensive shell analysis

### General Issues (All Files)
- **Security**: Hardcoded passwords, secrets, API keys
- **Code style**: Trailing whitespace, very long lines
- **File handling**: Issues reading or parsing files

### External Tool Integration
When available, the enhanced detector integrates with:
- **pylint**: Comprehensive Python analysis
- **flake8**: Python style and error checking
- **bandit**: Python security analysis
- **ESLint**: JavaScript/TypeScript linting
- **shellcheck**: Shell script analysis

## 📊 Report Formats

### Text Report (Default)
```
🔍 ENHANCED BUG DETECTION REPORT
============================================================
Total issues found: 15

Issues by severity:
  🔴 CRITICAL: 2
  🟠 HIGH: 3
  🟡 MEDIUM: 5
  🔵 LOW: 5

Issues by tool:
  🔧 custom: 8
  🔧 pylint: 4
  🔧 flake8: 3

📄 example.py
----------------------------------------

🔴 Line 12:0 - SecurityRisk [hardcoded-password] (custom)
   Potential hardcoded secret/password
   💡 Move secrets to environment variables or config files
```

### JSON Report
Structured data perfect for integration with other tools:
```json
{
  "total_issues": 15,
  "available_tools": ["pylint", "flake8"],
  "issues": [
    {
      "file_path": "example.py",
      "line_number": 12,
      "severity": "CRITICAL",
      "bug_type": "SecurityRisk",
      "message": "Potential hardcoded secret/password",
      "tool": "custom"
    }
  ]
}
```

### HTML Report
Beautiful web-based report with color-coded severity levels and interactive features.

## 🛠️ External Dependencies

### Required (Python)
- Python 3.6+
- Standard library modules only for basic functionality

### Optional (Enhanced Features)
- **PyYAML**: For YAML configuration files
- **pylint**: Advanced Python analysis
- **flake8**: Python style checking
- **bandit**: Security analysis for Python
- **mypy**: Type checking for Python
- **ESLint**: JavaScript/TypeScript analysis (requires Node.js)
- **shellcheck**: Shell script analysis

### Installation Commands
```bash
# Python tools
pip install pyyaml pylint flake8 bandit mypy

# JavaScript tools (requires Node.js/npm)
npm install -g eslint

# System tools
# Ubuntu/Debian: sudo apt-get install shellcheck
# macOS: brew install shellcheck
```

## 📁 Project Structure
```
.
├── bug_detector.py           # Core bug detection tool
├── enhanced_bug_detector.py  # Enhanced tool with external integration
├── setup.py                  # Setup and dependency installation
├── README.md                 # This documentation
└── examples/                 # Example files for testing
    ├── buggy_code.py        # Python file with intentional bugs
    ├── bad_script.sh        # Shell script with issues
    └── problematic.js       # JavaScript with problems
```

## 🎯 Examples

### Analyze a Python Project
```bash
# Basic analysis
python bug_detector.py my_project/ -r

# Enhanced analysis with external tools
python enhanced_bug_detector.py my_project/ -r -f html -o report.html

# Focus on security issues only
python enhanced_bug_detector.py my_project/ -r --severity HIGH
```

### Create and Use Configuration
```bash
# Create configuration
python enhanced_bug_detector.py --create-config .bugdetector.yaml

# Edit the configuration as needed
# nano .bugdetector.yaml

# Run with configuration
python enhanced_bug_detector.py . -r -c .bugdetector.yaml
```

### CI/CD Integration
```bash
# Exit with error code if critical issues found
python enhanced_bug_detector.py . -r --severity CRITICAL
if [ $? -ne 0 ]; then
    echo "Critical issues found! Build failed."
    exit 1
fi
```

## 🔧 Customization

### Adding New Language Support
The tools are designed to be extensible. To add support for a new language:

1. Create a new analyzer class inheriting from `CodeAnalyzer`
2. Implement language-specific analysis logic
3. Add the file extension mapping in the `BugDetector` class
4. Optionally integrate external tools for that language

### Custom Rules
You can create custom analysis rules by:
1. Modifying the configuration file
2. Adding new rule types in the analyzer classes
3. Implementing custom pattern matching

## 📈 Performance

- **Parallel processing**: Multiple files analyzed simultaneously
- **Timeout protection**: Individual file analysis won't hang the entire process
- **Memory efficient**: Streaming analysis for large files
- **Scalable**: Handles projects with thousands of files

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional language support
- More sophisticated analysis patterns
- Integration with additional external tools
- Performance optimizations
- UI improvements

## 📝 License

This project is provided as-is for educational and practical use. Feel free to modify and distribute according to your needs.

## 🐛 Bug Reports

If you find bugs in the bug detector (how meta!), please:
1. Create a minimal reproduction case
2. Include the command used and expected vs actual output
3. Specify your Python version and operating system

## 🚀 Roadmap

Future enhancements planned:
- **IDE integration**: VS Code extension, Vim plugin
- **Web interface**: Browser-based analysis and reporting
- **Machine learning**: AI-powered bug pattern detection
- **Team features**: Shared configurations, team dashboards
- **More languages**: Java, C++, Go, Rust support

---

Happy bug hunting! 🕷️