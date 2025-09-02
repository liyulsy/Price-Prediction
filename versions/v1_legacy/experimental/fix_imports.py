#!/usr/bin/env python3
"""
Script to fix import paths in training scripts after directory reorganization
"""

import os
import re

def fix_imports_in_file(filepath):
    """Fix import paths in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Add sys import if not present
    if 'import sys' not in content:
        content = re.sub(r'(import os)', r'\1\nimport sys', content)
    
    # Add sys.path.append if not present
    if 'sys.path.append' not in content:
        # Find the position after imports but before other code
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_pos = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break

        # Insert the sys.path.append line with __file__ check
        lines.insert(insert_pos, '')
        lines.insert(insert_pos + 1, "# Add the project root to Python path")
        lines.insert(insert_pos + 2, "current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()")
        lines.insert(insert_pos + 3, "project_root = os.path.join(current_dir, '..', '..')")
        lines.insert(insert_pos + 4, "sys.path.append(project_root)")
        lines.insert(insert_pos + 5, '')
        content = '\n'.join(lines)

    # Fix existing __file__ usage
    content = re.sub(
        r'sys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'.*?\)\)',
        "current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()\nproject_root = os.path.join(current_dir, '..', '..')\nsys.path.append(project_root)",
        content
    )
    
    # Fix specific import patterns
    replacements = [
        (r'from crypto_new_analyzer\.', 'from scripts.analysis.crypto_new_analyzer.'),
        (r'from analysis\.crypto_new_analyzer\.', 'from scripts.analysis.crypto_new_analyzer.'),
        # Keep models and dataloader imports as they are (they should work with sys.path.append)
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {filepath}")
        return True
    else:
        print(f"No changes needed in: {filepath}")
        return False

def main():
    """Main function to fix all training scripts"""
    training_dir = 'scripts/training'
    
    if not os.path.exists(training_dir):
        print(f"Directory {training_dir} not found!")
        return
    
    fixed_count = 0
    for filename in os.listdir(training_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(training_dir, filename)
            if fix_imports_in_file(filepath):
                fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files.")

if __name__ == '__main__':
    main()
