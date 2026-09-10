"""Static checks for the agent-authored scene.py contract."""

import ast


CONTRACT_NAMES = frozenset({
    "SCALE",
    "REFERENCE_FRAME",
    "JOINTS",
    "FRAMES",
})


class _ModuleBindingVisitor(ast.NodeVisitor):
    """Collect contract bindings without entering nested Python scopes."""

    def __init__(self):
        self.bindings = {name: [] for name in CONTRACT_NAMES}

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def _record(self, target):
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._record(item)
        elif isinstance(target, ast.Name) and target.id in CONTRACT_NAMES:
            self.bindings[target.id].append(target.lineno)

    def visit_Assign(self, node):
        for target in node.targets:
            self._record(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._record(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self._record(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node):
        self._record(node.target)
        self.visit(node.value)


def validate_scene_contract_source(source, path="scene.py"):
    """Reject repeated module-scope declarations in scene.py source."""
    tree = ast.parse(source, filename=path)
    visitor = _ModuleBindingVisitor()
    visitor.visit(tree)
    duplicates = {
        name: lines
        for name, lines in visitor.bindings.items()
        if len(lines) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"{name} at lines {', '.join(str(line) for line in lines)}"
            for name, lines in sorted(duplicates.items())
        )
        raise ValueError(
            f"{path}: duplicate scene contract declarations: {detail}. "
            "Keep exactly one module-scope declaration for each contract name.")
    return tree


def validate_scene_contract_file(path):
    """Read and validate one scene.py, returning its parsed AST."""
    with open(path, "r", encoding="utf-8") as f:
        return validate_scene_contract_source(f.read(), path=path)
