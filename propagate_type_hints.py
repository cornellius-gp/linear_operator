# Propagate type hints & signatures defined in _linear_operator.py to derived classes.
# Here we leverage libcst which can preserve the original whitespace & formatting of the file
# The idea is that we only want to change the type hints.
# This way we can enforce consistency between the base class signature and derived signatures.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import libcst as cst


class TypingCollector(cst.CSTVisitor):
    def __init__(self):
        # stack for storing the canonical name of the current function
        self.stack: List[Tuple[str, ...]] = []
        # store the annotations
        self.annotations: Dict[
            Tuple[str, ...],  # key: tuple of canonical class/function name
            Tuple[cst.Parameters, Optional[cst.Annotation]],  # value: (params, returns)
        ] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.stack.append(node.name.value)
        self.annotations[tuple(self.stack)] = (node.params, node.returns)
        return False  # pyi files don't support inner functions, return False to stop the traversal.

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.stack.pop()


class TypingTransformer(cst.CSTTransformer):

    # List of LinearOperator functions we do not want to propagate the signature from
    excluded_functions = ["__init__", "_check_args", "__torch_function__"]

    def __init__(self, annotations):
        # stack for storing the canonical name of the current function
        self.stack: List[Tuple[str, ...]] = []
        # store the annotations
        self.annotations: Dict[
            Tuple[str, ...],  # key: tuple of canonical class/function name
            Tuple[cst.Parameters, Optional[cst.Annotation]],  # value: (params, returns)
        ] = annotations

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        self.stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.stack.append(node.name.value)
        return False  # pyi files don't support inner functions, return False to stop the traversal.

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        key = tuple(self.stack)
        if key[-1] in TypingTransformer.excluded_functions:
            return updated_node
        key = ("LinearOperator", key[-1])
        self.stack.pop()
        if key in self.annotations:
            annotations = self.annotations[key]
            return updated_node.with_changes(params=annotations[0], returns=annotations[1])
        return updated_node


def collect_base_type_hints(base_filename: Path) -> TypingCollector:
    base_tree = cst.parse_module(base_filename.read_text())
    base_visitor = TypingCollector()
    base_tree.visit(base_visitor)
    return base_visitor


def copy_base_type_hints_to_derived(target: Path, base_visitor: TypingCollector) -> cst.Module:
    source_tree = cst.parse_module(target.read_text())
    transformer = TypingTransformer(base_visitor.annotations)
    modified_tree = source_tree.visit(transformer)
    return modified_tree


def main():
    directory = "linear_operator/operators"
    base_filename = Path(directory) / "_linear_operator.py"
    base_visitor = collect_base_type_hints(base_filename)

    pathlist = Path(directory).glob("*.py")
    for path in pathlist:
        if path.name[0] == "_":
            continue
        target = path
        target_out = Path(str(path) + "2")
        print(f"Processing {(target.name, target_out.name)}")
        modified_tree = copy_base_type_hints_to_derived(target, base_visitor)
        with open(target_out, "w") as f:
            f.write(modified_tree.code)


main()
