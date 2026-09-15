"""The teaching reference generator must use the teaching configuration."""

import ast
from pathlib import Path


def test_reference_generator_defaults_to_notebook_configuration():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'scripts/make_atm407_reference.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
             and node.args and isinstance(node.args[0], ast.Constant)
             and node.args[0].value == '--config']
    assert len(calls) == 1
    default = next(item.value for item in calls[0].keywords if item.arg == 'default')
    path = eval(compile(ast.Expression(default), '<config default>', 'eval'), {'root': root})
    assert path == root / 'scm/configs/atm407.toml'
    assert path.is_file()
