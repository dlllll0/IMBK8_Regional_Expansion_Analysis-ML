"""Read-only publication checks. Does not rerun private-data analysis."""
from pathlib import Path
import ast, json, re, sys


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    forbidden = {'.csv','.tsv','.xlsx','.xls','.pdf','.docx','.pptx','.zip','.parquet','.pkl','.joblib'}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(x in rel.parts for x in ['.git','.venv','private-review','outputs','__pycache__']):
            continue
        if path.suffix.lower() in forbidden and rel.as_posix() != 'docs/presentation-public.pdf':
            errors.append(f'private-like file: {rel}')
        if path.name == 'validate_repository.py':
            continue
        if path.suffix.lower() in {'.py','.md','.json','.ipynb','.txt'}:
            body = path.read_text(encoding='utf-8')
            if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}', body):
                errors.append(f'private path or secret pattern: {rel}')
            if path.suffix == '.py':
                ast.parse(body)
        if path.suffix == '.md':
            for dest in re.findall(r'\]\(([^)]+)\)', body):
                if '://' in dest or dest.startswith('#'):
                    continue
                if not (path.parent / dest.split('#',1)[0]).exists():
                    errors.append(f'broken link: {rel} -> {dest}')
        if path.suffix == '.ipynb':
            nb = json.loads(body)
            seen = set()
            for cell in nb['cells']:
                if cell.get('id') in seen or not cell.get('id'):
                    errors.append(f'missing or duplicate cell id: {rel}')
                seen.add(cell.get('id'))
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'stored output: {rel}')
                if cell.get('cell_type') == 'code':
                    ast.parse(''.join(cell.get('source', [])))
    summary = json.loads((root/'assets/stored-results.json').read_text(encoding='utf-8'))
    if len(summary['region_top3']) != 3:
        errors.append('top-three result missing')
    if errors:
        for error in errors: print('FAIL:', error)
        return 1
    print('PASS: syntax, links, clean notebook, no private-like files, aggregate result present.')
    print('NOTE: this check does not rerun private-data statistics or verify publication rights.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
