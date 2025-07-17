#!/usr/bin/env python3
"""
Comprehensive Bug Detection and Error Finding Tool
Supports multiple programming languages and various types of analysis.
"""

import os
import re
import ast
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import argparse


class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class BugReport:
    file_path: str
    line_number: int
    column: int
    severity: Severity
    bug_type: str
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


class CodeAnalyzer:
    """Base class for language-specific analyzers"""
    
    def __init__(self):
        self.bugs = []
    
    def analyze(self, file_path: str) -> List[BugReport]:
        """Analyze a file and return list of bugs found"""
        raise NotImplementedError
    
    def add_bug(self, file_path: str, line: int, col: int, severity: Severity, 
                bug_type: str, message: str, suggestion: str = None, snippet: str = None):
        """Helper method to add a bug report"""
        self.bugs.append(BugReport(
            file_path=file_path,
            line_number=line,
            column=col,
            severity=severity,
            bug_type=bug_type,
            message=message,
            suggestion=suggestion,
            code_snippet=snippet
        ))


class PythonAnalyzer(CodeAnalyzer):
    """Python-specific bug detection"""
    
    def analyze(self, file_path: str) -> List[BugReport]:
        self.bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path, lines)
            except SyntaxError as e:
                self.add_bug(file_path, e.lineno or 1, e.offset or 0, 
                           Severity.CRITICAL, "SyntaxError", str(e))
            
            # Text-based analysis
            self._analyze_text(file_path, lines)
            
        except Exception as e:
            self.add_bug(file_path, 1, 0, Severity.HIGH, "FileError", 
                        f"Could not analyze file: {str(e)}")
        
        return self.bugs
    
    def _analyze_ast(self, tree: ast.AST, file_path: str, lines: List[str]):
        """Analyze Python AST for bugs"""
        
        class BugVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.file_path = file_path
                self.lines = lines
                self.defined_vars = set()
                self.imported_names = set()
                self.function_calls = []
            
            def visit_Import(self, node):
                for alias in node.names:
                    self.imported_names.add(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    self.imported_names.add(alias.name)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    # Check for undefined variables
                    if (node.id not in self.defined_vars and 
                        node.id not in self.imported_names and
                        node.id not in dir(__builtins__)):
                        self.analyzer.add_bug(
                            self.file_path, node.lineno, node.col_offset,
                            Severity.HIGH, "NameError", 
                            f"Potentially undefined variable: '{node.id}'",
                            f"Check if '{node.id}' is defined before use"
                        )
                self.generic_visit(node)
            
            def visit_Call(self, node):
                self.function_calls.append(node)
                
                # Check for dangerous functions
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        self.analyzer.add_bug(
                            self.file_path, node.lineno, node.col_offset,
                            Severity.CRITICAL, "SecurityRisk",
                            "Use of 'eval()' is dangerous and should be avoided",
                            "Consider using safer alternatives like ast.literal_eval()"
                        )
                    elif node.func.id == 'exec':
                        self.analyzer.add_bug(
                            self.file_path, node.lineno, node.col_offset,
                            Severity.CRITICAL, "SecurityRisk",
                            "Use of 'exec()' is dangerous and should be avoided"
                        )
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        self.analyzer.add_bug(
                            self.file_path, handler.lineno, handler.col_offset,
                            Severity.MEDIUM, "BadPractice",
                            "Bare 'except:' clause catches all exceptions",
                            "Specify specific exception types to catch"
                        )
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                # Check for potential issues with comparisons
                if len(node.ops) > 1:
                    # Chained comparisons - check for common mistakes
                    for i, op in enumerate(node.ops):
                        if isinstance(op, ast.Is) and isinstance(node.comparators[i], ast.Constant):
                            if isinstance(node.comparators[i].value, (int, str, bool)):
                                self.analyzer.add_bug(
                                    self.file_path, node.lineno, node.col_offset,
                                    Severity.MEDIUM, "BadPractice",
                                    "Use '==' instead of 'is' for value comparison",
                                    "Use 'is' only for identity comparison (None, True, False)"
                                )
                self.generic_visit(node)
        
        visitor = BugVisitor(self)
        visitor.visit(tree)
    
    def _analyze_text(self, file_path: str, lines: List[str]):
        """Text-based analysis for common Python issues"""
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for common issues
            if 'print(' in line and not line_stripped.startswith('#'):
                # Potential debug print statements
                self.add_bug(file_path, i, line.find('print('), Severity.LOW,
                           "CodeQuality", "Debug print statement found",
                           "Remove debug prints before production")
            
            # Check for TODO/FIXME comments
            if re.search(r'#.*\b(TODO|FIXME|HACK|BUG)\b', line, re.IGNORECASE):
                self.add_bug(file_path, i, 0, Severity.LOW,
                           "TODO", "TODO/FIXME comment found",
                           "Address this TODO item")
            
            # Check for long lines
            if len(line) > 120:
                self.add_bug(file_path, i, 120, Severity.LOW,
                           "CodeStyle", f"Line too long ({len(line)} characters)",
                           "Break long lines for better readability")
            
            # Check for potential SQL injection
            if re.search(r'(execute|query)\s*\(\s*["\'].*%.*["\']', line):
                self.add_bug(file_path, i, 0, Severity.HIGH,
                           "SecurityRisk", "Potential SQL injection vulnerability",
                           "Use parameterized queries instead of string formatting")


class JavaScriptAnalyzer(CodeAnalyzer):
    """JavaScript/TypeScript bug detection"""
    
    def analyze(self, file_path: str) -> List[BugReport]:
        self.bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self._analyze_text(file_path, lines)
            
        except Exception as e:
            self.add_bug(file_path, 1, 0, Severity.HIGH, "FileError",
                        f"Could not analyze file: {str(e)}")
        
        return self.bugs
    
    def _analyze_text(self, file_path: str, lines: List[str]):
        """Text-based analysis for JavaScript issues"""
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for == instead of ===
            if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                if re.search(r'[^!<>=]==[^=]', line):
                    self.add_bug(file_path, i, line.find('=='), Severity.MEDIUM,
                               "BadPractice", "Use strict equality (===) instead of ==",
                               "Replace == with === for type-safe comparison")
            
            # Check for var usage
            if re.search(r'\bvar\s+\w+', line):
                self.add_bug(file_path, i, line.find('var'), Severity.LOW,
                           "ModernJS", "Use 'let' or 'const' instead of 'var'",
                           "var has function scope, let/const have block scope")
            
            # Check for console.log
            if 'console.log' in line and not line_stripped.startswith('//'):
                self.add_bug(file_path, i, line.find('console.log'), Severity.LOW,
                           "CodeQuality", "Debug console.log found",
                           "Remove debug statements before production")
            
            # Check for eval usage
            if 'eval(' in line:
                self.add_bug(file_path, i, line.find('eval('), Severity.CRITICAL,
                           "SecurityRisk", "Use of eval() is dangerous",
                           "Avoid eval() - use safer alternatives")
            
            # Check for potential XSS
            if re.search(r'innerHTML\s*=.*\+', line):
                self.add_bug(file_path, i, 0, Severity.HIGH,
                           "SecurityRisk", "Potential XSS vulnerability with innerHTML",
                           "Sanitize user input or use textContent instead")


class GeneralAnalyzer(CodeAnalyzer):
    """General analysis for any code file"""
    
    def analyze(self, file_path: str) -> List[BugReport]:
        self.bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self._analyze_general(file_path, lines)
            
        except Exception as e:
            self.add_bug(file_path, 1, 0, Severity.HIGH, "FileError",
                        f"Could not analyze file: {str(e)}")
        
        return self.bugs
    
    def _analyze_general(self, file_path: str, lines: List[str]):
        """General analysis applicable to any programming language"""
        
        for i, line in enumerate(lines, 1):
            # Check for potential passwords/secrets
            if re.search(r'(password|secret|key|token)\s*[=:]\s*["\'][^"\']{8,}["\']', line, re.IGNORECASE):
                self.add_bug(file_path, i, 0, Severity.CRITICAL,
                           "SecurityRisk", "Potential hardcoded secret/password",
                           "Move secrets to environment variables or config files")
            
            # Check for very long lines
            if len(line) > 150:
                self.add_bug(file_path, i, 150, Severity.LOW,
                           "CodeStyle", f"Very long line ({len(line)} characters)",
                           "Consider breaking long lines for readability")
            
            # Check for trailing whitespace
            if line.rstrip() != line.rstrip('\n'):
                self.add_bug(file_path, i, len(line.rstrip()), Severity.LOW,
                           "CodeStyle", "Trailing whitespace",
                           "Remove trailing whitespace")


class BugDetector:
    """Main bug detection engine"""
    
    def __init__(self):
        self.analyzers = {
            '.py': PythonAnalyzer(),
            '.js': JavaScriptAnalyzer(),
            '.ts': JavaScriptAnalyzer(),
            '.jsx': JavaScriptAnalyzer(),
            '.tsx': JavaScriptAnalyzer(),
        }
        self.general_analyzer = GeneralAnalyzer()
    
    def analyze_file(self, file_path: str) -> List[BugReport]:
        """Analyze a single file"""
        ext = Path(file_path).suffix.lower()
        
        if ext in self.analyzers:
            analyzer = self.analyzers[ext]
        else:
            analyzer = self.general_analyzer
        
        return analyzer.analyze(file_path)
    
    def analyze_directory(self, directory: str, recursive: bool = True) -> List[BugReport]:
        """Analyze all files in a directory"""
        all_bugs = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip common directories to ignore
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
                
                for file in files:
                    if self._should_analyze_file(file):
                        file_path = os.path.join(root, file)
                        bugs = self.analyze_file(file_path)
                        all_bugs.extend(bugs)
        else:
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)) and self._should_analyze_file(file):
                    file_path = os.path.join(directory, file)
                    bugs = self.analyze_file(file_path)
                    all_bugs.extend(bugs)
        
        return all_bugs
    
    def _should_analyze_file(self, filename: str) -> bool:
        """Determine if a file should be analyzed"""
        # Skip binary files, hidden files, and common non-code files
        skip_extensions = {'.pyc', '.pyo', '.class', '.jar', '.war', '.exe', '.dll', '.so', 
                          '.img', '.jpg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz'}
        skip_patterns = ['node_modules', '__pycache__', '.git', '.DS_Store']
        
        ext = Path(filename).suffix.lower()
        if ext in skip_extensions:
            return False
        
        if any(pattern in filename for pattern in skip_patterns):
            return False
        
        return True
    
    def generate_report(self, bugs: List[BugReport], format: str = 'text') -> str:
        """Generate a formatted report of bugs"""
        
        if not bugs:
            return "✅ No bugs detected! Your code looks clean."
        
        # Sort bugs by severity and file
        bugs.sort(key=lambda x: (x.severity.value, x.file_path, x.line_number))
        
        if format == 'json':
            return self._generate_json_report(bugs)
        else:
            return self._generate_text_report(bugs)
    
    def _generate_text_report(self, bugs: List[BugReport]) -> str:
        """Generate text format report"""
        report = []
        report.append("🐛 BUG DETECTION REPORT")
        report.append("=" * 50)
        report.append(f"Total issues found: {len(bugs)}")
        
        # Count by severity
        severity_counts = {}
        for bug in bugs:
            severity_counts[bug.severity] = severity_counts.get(bug.severity, 0) + 1
        
        report.append("\nIssues by severity:")
        for severity in Severity:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
                report.append(f"  {emoji.get(severity.value, '⚪')} {severity.value}: {count}")
        
        report.append("\n" + "=" * 50)
        
        current_file = None
        for bug in bugs:
            if bug.file_path != current_file:
                current_file = bug.file_path
                report.append(f"\n📄 {current_file}")
                report.append("-" * 30)
            
            severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
            emoji = severity_emoji.get(bug.severity.value, "⚪")
            
            report.append(f"\n{emoji} Line {bug.line_number}:{bug.column} - {bug.bug_type}")
            report.append(f"   {bug.message}")
            if bug.suggestion:
                report.append(f"   💡 {bug.suggestion}")
            if bug.code_snippet:
                report.append(f"   📝 {bug.code_snippet}")
        
        return "\n".join(report)
    
    def _generate_json_report(self, bugs: List[BugReport]) -> str:
        """Generate JSON format report"""
        bugs_data = []
        for bug in bugs:
            bugs_data.append({
                'file_path': bug.file_path,
                'line_number': bug.line_number,
                'column': bug.column,
                'severity': bug.severity.value,
                'bug_type': bug.bug_type,
                'message': bug.message,
                'suggestion': bug.suggestion,
                'code_snippet': bug.code_snippet
            })
        
        return json.dumps({
            'total_issues': len(bugs),
            'issues': bugs_data
        }, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Bug Detection Tool')
    parser.add_argument('path', help='File or directory to analyze')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Recursively analyze directories')
    parser.add_argument('-f', '--format', choices=['text', 'json'], default='text',
                       help='Output format')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--severity', choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                       help='Minimum severity level to report')
    
    args = parser.parse_args()
    
    detector = BugDetector()
    
    if os.path.isfile(args.path):
        bugs = detector.analyze_file(args.path)
    elif os.path.isdir(args.path):
        bugs = detector.analyze_directory(args.path, args.recursive)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)
    
    # Filter by severity if specified
    if args.severity:
        min_severity = Severity(args.severity)
        severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        min_level = severity_order[min_severity]
        bugs = [bug for bug in bugs if severity_order[bug.severity] >= min_level]
    
    # Generate report
    report = detector.generate_report(bugs, args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Exit with error code if critical issues found
    critical_bugs = [bug for bug in bugs if bug.severity == Severity.CRITICAL]
    if critical_bugs:
        sys.exit(1)


if __name__ == '__main__':
    main()