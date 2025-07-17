#!/usr/bin/env python3
"""
Enhanced Bug Detection Tool with External Tool Integration
Integrates with popular linters and static analysis tools.
"""

import os
import re
import ast
import json
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    tool: str = "custom"
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    rule_id: Optional[str] = None


class ExternalToolIntegrator:
    """Integrates with external static analysis tools"""
    
    def __init__(self):
        self.available_tools = self._check_available_tools()
    
    def _check_available_tools(self) -> Set[str]:
        """Check which external tools are available"""
        tools = set()
        
        # Check for Python tools
        for tool in ['pylint', 'flake8', 'mypy', 'bandit', 'black', 'isort']:
            if shutil.which(tool):
                tools.add(tool)
        
        # Check for JavaScript tools
        for tool in ['eslint', 'jshint', 'prettier']:
            if shutil.which(tool):
                tools.add(tool)
        
        # Check for other tools
        for tool in ['shellcheck', 'hadolint', 'yamllint']:
            if shutil.which(tool):
                tools.add(tool)
        
        return tools
    
    def run_pylint(self, file_path: str) -> List[BugReport]:
        """Run pylint on Python files"""
        if 'pylint' not in self.available_tools:
            return []
        
        bugs = []
        try:
            result = subprocess.run(
                ['pylint', '--output-format=json', '--reports=no', file_path],
                capture_output=True, text=True, timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for item in data:
                    severity = self._map_pylint_severity(item.get('type', 'info'))
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=item.get('line', 1),
                        column=item.get('column', 0),
                        severity=severity,
                        bug_type=f"pylint-{item.get('type', 'unknown')}",
                        message=item.get('message', ''),
                        tool='pylint',
                        rule_id=item.get('symbol', '')
                    ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        
        return bugs
    
    def run_flake8(self, file_path: str) -> List[BugReport]:
        """Run flake8 on Python files"""
        if 'flake8' not in self.available_tools:
            return []
        
        bugs = []
        try:
            result = subprocess.run(
                ['flake8', '--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s', file_path],
                capture_output=True, text=True, timeout=30
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':', 4)
                    if len(parts) >= 5:
                        bugs.append(BugReport(
                            file_path=file_path,
                            line_number=int(parts[1]),
                            column=int(parts[2]),
                            severity=self._map_flake8_severity(parts[3]),
                            bug_type=f"flake8-{parts[3]}",
                            message=parts[4],
                            tool='flake8',
                            rule_id=parts[3]
                        ))
        except (subprocess.TimeoutExpired, ValueError, Exception):
            pass
        
        return bugs
    
    def run_bandit(self, file_path: str) -> List[BugReport]:
        """Run bandit security analysis on Python files"""
        if 'bandit' not in self.available_tools:
            return []
        
        bugs = []
        try:
            result = subprocess.run(
                ['bandit', '-f', 'json', '-q', file_path],
                capture_output=True, text=True, timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for item in data.get('results', []):
                    severity = self._map_bandit_severity(item.get('issue_severity', 'LOW'))
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=item.get('line_number', 1),
                        column=item.get('col_offset', 0),
                        severity=severity,
                        bug_type=f"security-{item.get('test_name', 'unknown')}",
                        message=item.get('issue_text', ''),
                        tool='bandit',
                        rule_id=item.get('test_id', ''),
                        suggestion=item.get('more_info', '')
                    ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        
        return bugs
    
    def run_eslint(self, file_path: str) -> List[BugReport]:
        """Run ESLint on JavaScript/TypeScript files"""
        if 'eslint' not in self.available_tools:
            return []
        
        bugs = []
        try:
            result = subprocess.run(
                ['eslint', '--format=json', file_path],
                capture_output=True, text=True, timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for file_result in data:
                    for message in file_result.get('messages', []):
                        severity = self._map_eslint_severity(message.get('severity', 1))
                        bugs.append(BugReport(
                            file_path=file_path,
                            line_number=message.get('line', 1),
                            column=message.get('column', 0),
                            severity=severity,
                            bug_type=f"eslint-{message.get('ruleId', 'unknown')}",
                            message=message.get('message', ''),
                            tool='eslint',
                            rule_id=message.get('ruleId', '')
                        ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        
        return bugs
    
    def run_shellcheck(self, file_path: str) -> List[BugReport]:
        """Run shellcheck on shell scripts"""
        if 'shellcheck' not in self.available_tools:
            return []
        
        bugs = []
        try:
            result = subprocess.run(
                ['shellcheck', '--format=json', file_path],
                capture_output=True, text=True, timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                for item in data:
                    severity = self._map_shellcheck_severity(item.get('level', 'info'))
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=item.get('line', 1),
                        column=item.get('column', 0),
                        severity=severity,
                        bug_type=f"shellcheck-{item.get('code', 'unknown')}",
                        message=item.get('message', ''),
                        tool='shellcheck',
                        rule_id=f"SC{item.get('code', '')}"
                    ))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        
        return bugs
    
    def _map_pylint_severity(self, pylint_type: str) -> Severity:
        """Map pylint message types to our severity levels"""
        mapping = {
            'error': Severity.HIGH,
            'warning': Severity.MEDIUM,
            'refactor': Severity.LOW,
            'convention': Severity.LOW,
            'info': Severity.LOW
        }
        return mapping.get(pylint_type.lower(), Severity.MEDIUM)
    
    def _map_flake8_severity(self, code: str) -> Severity:
        """Map flake8 error codes to severity levels"""
        if code.startswith('E9') or code.startswith('F'):
            return Severity.HIGH
        elif code.startswith('E') or code.startswith('W'):
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _map_bandit_severity(self, bandit_severity: str) -> Severity:
        """Map bandit severity to our levels"""
        mapping = {
            'HIGH': Severity.CRITICAL,
            'MEDIUM': Severity.HIGH,
            'LOW': Severity.MEDIUM
        }
        return mapping.get(bandit_severity.upper(), Severity.MEDIUM)
    
    def _map_eslint_severity(self, eslint_severity: int) -> Severity:
        """Map ESLint severity to our levels"""
        if eslint_severity == 2:
            return Severity.HIGH
        elif eslint_severity == 1:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _map_shellcheck_severity(self, level: str) -> Severity:
        """Map shellcheck levels to our severity"""
        mapping = {
            'error': Severity.HIGH,
            'warning': Severity.MEDIUM,
            'info': Severity.LOW,
            'style': Severity.LOW
        }
        return mapping.get(level.lower(), Severity.MEDIUM)


class ConfigurableRuleEngine:
    """Manages configurable analysis rules"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'rules': {
                'max_line_length': 120,
                'check_debug_statements': True,
                'check_todos': True,
                'check_security_issues': True,
                'check_code_style': True,
                'check_complexity': True,
                'max_complexity': 10
            },
            'severity_overrides': {},
            'ignore_patterns': [
                '*.pyc', '*.pyo', '__pycache__/*', 'node_modules/*',
                '.git/*', '*.min.js', '*.bundle.js'
            ],
            'external_tools': {
                'enabled': True,
                'python': ['pylint', 'flake8', 'bandit'],
                'javascript': ['eslint'],
                'shell': ['shellcheck']
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge with defaults
                default_config.update(user_config)
            except Exception:
                pass
        
        return default_config
    
    def should_check_rule(self, rule_name: str) -> bool:
        """Check if a rule should be applied"""
        return self.config.get('rules', {}).get(rule_name, True)
    
    def get_rule_value(self, rule_name: str, default: Any = None) -> Any:
        """Get a rule configuration value"""
        return self.config.get('rules', {}).get(rule_name, default)
    
    def get_severity_override(self, rule_id: str) -> Optional[Severity]:
        """Get severity override for a specific rule"""
        override = self.config.get('severity_overrides', {}).get(rule_id)
        if override:
            try:
                return Severity(override.upper())
            except ValueError:
                pass
        return None


class EnhancedBugDetector:
    """Enhanced bug detector with external tool integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.rule_engine = ConfigurableRuleEngine(config_path)
        self.external_tools = ExternalToolIntegrator()
        self.custom_analyzers = {
            '.py': self._analyze_python_custom,
            '.js': self._analyze_javascript_custom,
            '.ts': self._analyze_javascript_custom,
            '.jsx': self._analyze_javascript_custom,
            '.tsx': self._analyze_javascript_custom,
            '.sh': self._analyze_shell_custom,
            '.bash': self._analyze_shell_custom,
        }
    
    def analyze_file(self, file_path: str) -> List[BugReport]:
        """Analyze a single file with both custom and external tools"""
        all_bugs = []
        ext = Path(file_path).suffix.lower()
        
        # Run custom analysis
        if ext in self.custom_analyzers:
            custom_bugs = self._run_with_timeout(
                self.custom_analyzers[ext], file_path, timeout=30
            )
            all_bugs.extend(custom_bugs)
        else:
            custom_bugs = self._analyze_general(file_path)
            all_bugs.extend(custom_bugs)
        
        # Run external tools if enabled
        if self.rule_engine.config.get('external_tools', {}).get('enabled', True):
            external_bugs = self._run_external_tools(file_path, ext)
            all_bugs.extend(external_bugs)
        
        # Apply severity overrides
        for bug in all_bugs:
            if bug.rule_id:
                override = self.rule_engine.get_severity_override(bug.rule_id)
                if override:
                    bug.severity = override
        
        return all_bugs
    
    def _run_with_timeout(self, func, *args, timeout=30):
        """Run function with timeout"""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args)
                return future.result(timeout=timeout)
        except Exception:
            return []
    
    def _run_external_tools(self, file_path: str, ext: str) -> List[BugReport]:
        """Run appropriate external tools based on file type"""
        bugs = []
        
        if ext == '.py':
            tools = self.rule_engine.config.get('external_tools', {}).get('python', [])
            for tool in tools:
                if tool == 'pylint':
                    bugs.extend(self.external_tools.run_pylint(file_path))
                elif tool == 'flake8':
                    bugs.extend(self.external_tools.run_flake8(file_path))
                elif tool == 'bandit':
                    bugs.extend(self.external_tools.run_bandit(file_path))
        
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            tools = self.rule_engine.config.get('external_tools', {}).get('javascript', [])
            for tool in tools:
                if tool == 'eslint':
                    bugs.extend(self.external_tools.run_eslint(file_path))
        
        elif ext in ['.sh', '.bash']:
            tools = self.rule_engine.config.get('external_tools', {}).get('shell', [])
            for tool in tools:
                if tool == 'shellcheck':
                    bugs.extend(self.external_tools.run_shellcheck(file_path))
        
        return bugs
    
    def _analyze_python_custom(self, file_path: str) -> List[BugReport]:
        """Custom Python analysis with configurable rules"""
        bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # AST analysis
            try:
                tree = ast.parse(content)
                bugs.extend(self._analyze_python_ast(tree, file_path, lines))
            except SyntaxError as e:
                bugs.append(BugReport(
                    file_path=file_path,
                    line_number=e.lineno or 1,
                    column=e.offset or 0,
                    severity=Severity.CRITICAL,
                    bug_type="SyntaxError",
                    message=str(e),
                    tool="custom"
                ))
            
            # Text-based analysis
            bugs.extend(self._analyze_python_text(file_path, lines))
            
        except Exception as e:
            bugs.append(BugReport(
                file_path=file_path,
                line_number=1,
                column=0,
                severity=Severity.HIGH,
                bug_type="FileError",
                message=f"Could not analyze file: {str(e)}",
                tool="custom"
            ))
        
        return bugs
    
    def _analyze_python_ast(self, tree: ast.AST, file_path: str, lines: List[str]) -> List[BugReport]:
        """Analyze Python AST with enhanced checks"""
        bugs = []
        
        class EnhancedBugVisitor(ast.NodeVisitor):
            def __init__(self):
                self.bugs = []
                self.complexity = 0
                self.depth = 0
                self.max_depth = 0
                self.functions = []
                self.classes = []
            
            def visit_FunctionDef(self, node):
                self.functions.append(node)
                old_complexity = self.complexity
                self.complexity = 1  # Reset for function
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                
                self.generic_visit(node)
                
                # Check complexity
                if (self.complexity > self.rule_engine.get_rule_value('max_complexity', 10) and
                    self.rule_engine.should_check_rule('check_complexity')):
                    self.bugs.append(BugReport(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity=Severity.MEDIUM,
                        bug_type="Complexity",
                        message=f"Function '{node.name}' has high complexity ({self.complexity})",
                        tool="custom",
                        suggestion="Consider breaking this function into smaller functions"
                    ))
                
                self.complexity = old_complexity
                self.depth -= 1
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            # Add other complexity-increasing nodes...
        
        visitor = EnhancedBugVisitor()
        visitor.rule_engine = self.rule_engine
        visitor.visit(tree)
        bugs.extend(visitor.bugs)
        
        return bugs
    
    def _analyze_python_text(self, file_path: str, lines: List[str]) -> List[BugReport]:
        """Text-based Python analysis"""
        bugs = []
        max_line_length = self.rule_engine.get_rule_value('max_line_length', 120)
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > max_line_length:
                bugs.append(BugReport(
                    file_path=file_path,
                    line_number=i,
                    column=max_line_length,
                    severity=Severity.LOW,
                    bug_type="LineLength",
                    message=f"Line too long ({len(line)} > {max_line_length} characters)",
                    tool="custom",
                    suggestion="Break long lines for better readability"
                ))
            
            # Check for debug statements
            if (self.rule_engine.should_check_rule('check_debug_statements') and
                'print(' in line and not line.strip().startswith('#')):
                bugs.append(BugReport(
                    file_path=file_path,
                    line_number=i,
                    column=line.find('print('),
                    severity=Severity.LOW,
                    bug_type="DebugStatement",
                    message="Debug print statement found",
                    tool="custom",
                    suggestion="Remove debug prints before production"
                ))
            
            # Check for TODOs
            if (self.rule_engine.should_check_rule('check_todos') and
                re.search(r'#.*\b(TODO|FIXME|HACK|BUG)\b', line, re.IGNORECASE)):
                bugs.append(BugReport(
                    file_path=file_path,
                    line_number=i,
                    column=0,
                    severity=Severity.LOW,
                    bug_type="TODO",
                    message="TODO/FIXME comment found",
                    tool="custom",
                    suggestion="Address this TODO item"
                ))
        
        return bugs
    
    def _analyze_javascript_custom(self, file_path: str) -> List[BugReport]:
        """Custom JavaScript/TypeScript analysis"""
        bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            bugs.extend(self._analyze_javascript_text(file_path, lines))
            
        except Exception as e:
            bugs.append(BugReport(
                file_path=file_path,
                line_number=1,
                column=0,
                severity=Severity.HIGH,
                bug_type="FileError",
                message=f"Could not analyze file: {str(e)}",
                tool="custom"
            ))
        
        return bugs
    
    def _analyze_javascript_text(self, file_path: str, lines: List[str]) -> List[BugReport]:
        """Text-based JavaScript analysis"""
        bugs = []
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for == instead of ===
            if '==' in line and '===' not in line:
                if re.search(r'[^!<>=]==[^=]', line):
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=i,
                        column=line.find('=='),
                        severity=Severity.MEDIUM,
                        bug_type="EqualityCheck",
                        message="Use strict equality (===) instead of ==",
                        tool="custom",
                        suggestion="Replace == with === for type-safe comparison"
                    ))
            
            # Check for console.log
            if ('console.log' in line and not line_stripped.startswith('//') and
                self.rule_engine.should_check_rule('check_debug_statements')):
                bugs.append(BugReport(
                    file_path=file_path,
                    line_number=i,
                    column=line.find('console.log'),
                    severity=Severity.LOW,
                    bug_type="DebugStatement",
                    message="Debug console.log found",
                    tool="custom",
                    suggestion="Remove debug statements before production"
                ))
        
        return bugs
    
    def _analyze_shell_custom(self, file_path: str) -> List[BugReport]:
        """Custom shell script analysis"""
        bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for unquoted variables
                if re.search(r'\$\w+(?!["\'\)])', line) and not line_stripped.startswith('#'):
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=i,
                        column=0,
                        severity=Severity.MEDIUM,
                        bug_type="UnquotedVariable",
                        message="Unquoted variable usage detected",
                        tool="custom",
                        suggestion="Quote variables to prevent word splitting"
                    ))
                
                # Check for missing set -e
                if i == 1 and line.startswith('#!') and 'set -e' not in lines[:5]:
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=i,
                        column=0,
                        severity=Severity.LOW,
                        bug_type="MissingSetE",
                        message="Consider using 'set -e' for better error handling",
                        tool="custom",
                        suggestion="Add 'set -e' near the top of the script"
                    ))
        
        except Exception as e:
            bugs.append(BugReport(
                file_path=file_path,
                line_number=1,
                column=0,
                severity=Severity.HIGH,
                bug_type="FileError",
                message=f"Could not analyze file: {str(e)}",
                tool="custom"
            ))
        
        return bugs
    
    def _analyze_general(self, file_path: str) -> List[BugReport]:
        """General analysis for any file type"""
        bugs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                # Check for secrets
                if (self.rule_engine.should_check_rule('check_security_issues') and
                    re.search(r'(password|secret|key|token)\s*[=:]\s*["\'][^"\']{8,}["\']', line, re.IGNORECASE)):
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=i,
                        column=0,
                        severity=Severity.CRITICAL,
                        bug_type="HardcodedSecret",
                        message="Potential hardcoded secret/password",
                        tool="custom",
                        suggestion="Move secrets to environment variables or config files"
                    ))
                
                # Check for trailing whitespace
                if (self.rule_engine.should_check_rule('check_code_style') and
                    line.rstrip() != line.rstrip('\n')):
                    bugs.append(BugReport(
                        file_path=file_path,
                        line_number=i,
                        column=len(line.rstrip()),
                        severity=Severity.LOW,
                        bug_type="TrailingWhitespace",
                        message="Trailing whitespace",
                        tool="custom",
                        suggestion="Remove trailing whitespace"
                    ))
        
        except Exception as e:
            bugs.append(BugReport(
                file_path=file_path,
                line_number=1,
                column=0,
                severity=Severity.HIGH,
                bug_type="FileError",
                message=f"Could not analyze file: {str(e)}",
                tool="custom"
            ))
        
        return bugs
    
    def analyze_directory(self, directory: str, recursive: bool = True) -> List[BugReport]:
        """Analyze directory with parallel processing"""
        all_bugs = []
        files_to_analyze = []
        
        ignore_patterns = self.rule_engine.config.get('ignore_patterns', [])
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if not any(
                    re.match(pattern.replace('*', '.*'), d) for pattern in ignore_patterns
                )]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if not any(re.match(pattern.replace('*', '.*'), file_path) for pattern in ignore_patterns):
                        files_to_analyze.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if (os.path.isfile(file_path) and 
                    not any(re.match(pattern.replace('*', '.*'), file_path) for pattern in ignore_patterns)):
                    files_to_analyze.append(file_path)
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(self.analyze_file, file_path): file_path 
                             for file_path in files_to_analyze}
            
            for future in as_completed(future_to_file):
                try:
                    bugs = future.result(timeout=60)
                    all_bugs.extend(bugs)
                except Exception:
                    # If analysis fails, continue with other files
                    pass
        
        return all_bugs
    
    def generate_report(self, bugs: List[BugReport], format: str = 'text') -> str:
        """Generate enhanced report with tool information"""
        if not bugs:
            return "✅ No bugs detected! Your code looks clean."
        
        if format == 'json':
            return self._generate_json_report(bugs)
        elif format == 'html':
            return self._generate_html_report(bugs)
        else:
            return self._generate_text_report(bugs)
    
    def _generate_text_report(self, bugs: List[BugReport]) -> str:
        """Generate enhanced text report"""
        bugs.sort(key=lambda x: (x.severity.value, x.file_path, x.line_number))
        
        report = []
        report.append("🔍 ENHANCED BUG DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Total issues found: {len(bugs)}")
        
        # Count by severity and tool
        severity_counts = {}
        tool_counts = {}
        
        for bug in bugs:
            severity_counts[bug.severity] = severity_counts.get(bug.severity, 0) + 1
            tool_counts[bug.tool] = tool_counts.get(bug.tool, 0) + 1
        
        report.append("\nIssues by severity:")
        for severity in Severity:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
                report.append(f"  {emoji.get(severity.value, '⚪')} {severity.value}: {count}")
        
        report.append("\nIssues by tool:")
        for tool, count in tool_counts.items():
            report.append(f"  🔧 {tool}: {count}")
        
        # Available external tools
        available_tools = list(self.external_tools.available_tools)
        if available_tools:
            report.append(f"\nAvailable external tools: {', '.join(available_tools)}")
        
        report.append("\n" + "=" * 60)
        
        current_file = None
        for bug in bugs:
            if bug.file_path != current_file:
                current_file = bug.file_path
                report.append(f"\n📄 {current_file}")
                report.append("-" * 40)
            
            severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
            emoji = severity_emoji.get(bug.severity.value, "⚪")
            
            rule_info = f" [{bug.rule_id}]" if bug.rule_id else ""
            tool_info = f" ({bug.tool})"
            
            report.append(f"\n{emoji} Line {bug.line_number}:{bug.column} - {bug.bug_type}{rule_info}{tool_info}")
            report.append(f"   {bug.message}")
            if bug.suggestion:
                report.append(f"   💡 {bug.suggestion}")
            if bug.code_snippet:
                report.append(f"   📝 {bug.code_snippet}")
        
        return "\n".join(report)
    
    def _generate_json_report(self, bugs: List[BugReport]) -> str:
        """Generate JSON report"""
        bugs_data = [asdict(bug) for bug in bugs]
        
        # Convert Severity enum to string
        for bug_data in bugs_data:
            bug_data['severity'] = bug_data['severity'].value if hasattr(bug_data['severity'], 'value') else bug_data['severity']
        
        return json.dumps({
            'total_issues': len(bugs),
            'available_tools': list(self.external_tools.available_tools),
            'config': self.rule_engine.config,
            'issues': bugs_data
        }, indent=2, default=str)
    
    def _generate_html_report(self, bugs: List[BugReport]) -> str:
        """Generate HTML report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bug Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .severity-critical { color: #d32f2f; }
                .severity-high { color: #f57c00; }
                .severity-medium { color: #fbc02d; }
                .severity-low { color: #1976d2; }
                .bug-item { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
                .file-header { font-weight: bold; margin-top: 20px; }
                .suggestion { font-style: italic; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Bug Detection Report</h1>
                <p>Total issues found: """ + str(len(bugs)) + """</p>
            </div>
        """
        
        current_file = None
        for bug in bugs:
            if bug.file_path != current_file:
                current_file = bug.file_path
                html += f'<div class="file-header">📄 {current_file}</div>'
            
            severity_class = f"severity-{bug.severity.value.lower()}"
            html += f'''
            <div class="bug-item {severity_class}">
                <strong>Line {bug.line_number}:{bug.column} - {bug.bug_type}</strong>
                {f" [{bug.rule_id}]" if bug.rule_id else ""} ({bug.tool})
                <br>{bug.message}
                {f'<div class="suggestion">💡 {bug.suggestion}</div>' if bug.suggestion else ""}
            </div>
            '''
        
        html += """
        </body>
        </html>
        """
        
        return html


def create_default_config(config_path: str):
    """Create a default configuration file"""
    default_config = {
        'rules': {
            'max_line_length': 120,
            'check_debug_statements': True,
            'check_todos': True,
            'check_security_issues': True,
            'check_code_style': True,
            'check_complexity': True,
            'max_complexity': 10
        },
        'severity_overrides': {
            'E501': 'LOW',  # Line too long - flake8
            'W503': 'LOW',  # Line break before binary operator
        },
        'ignore_patterns': [
            '*.pyc', '*.pyo', '__pycache__/*', 'node_modules/*',
            '.git/*', '*.min.js', '*.bundle.js', 'dist/*', 'build/*'
        ],
        'external_tools': {
            'enabled': True,
            'python': ['pylint', 'flake8', 'bandit'],
            'javascript': ['eslint'],
            'shell': ['shellcheck']
        }
    }
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(default_config, f, default_flow_style=False)
        else:
            json.dump(default_config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Bug Detection Tool')
    parser.add_argument('path', help='File or directory to analyze')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Recursively analyze directories')
    parser.add_argument('-f', '--format', choices=['text', 'json', 'html'], default='text',
                       help='Output format')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--create-config', help='Create default configuration file')
    parser.add_argument('--severity', choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                       help='Minimum severity level to report')
    parser.add_argument('--tools', help='Show available external tools and exit',
                       action='store_true')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.create_config)
        print(f"Default configuration created at {args.create_config}")
        return
    
    detector = EnhancedBugDetector(args.config)
    
    if args.tools:
        print("Available external tools:")
        for tool in detector.external_tools.available_tools:
            print(f"  ✅ {tool}")
        
        missing_tools = {
            'pylint', 'flake8', 'bandit', 'mypy',
            'eslint', 'shellcheck'
        } - detector.external_tools.available_tools
        
        if missing_tools:
            print("\nMissing tools (install for enhanced analysis):")
            for tool in missing_tools:
                print(f"  ❌ {tool}")
        return
    
    if os.path.isfile(args.path):
        bugs = detector.analyze_file(args.path)
    elif os.path.isdir(args.path):
        bugs = detector.analyze_directory(args.path, args.recursive)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)
    
    # Filter by severity
    if args.severity:
        min_severity = Severity(args.severity)
        severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        min_level = severity_order[min_severity]
        bugs = [bug for bug in bugs if severity_order[bug.severity] >= min_level]
    
    # Generate report
    report = detector.generate_report(bugs, args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
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