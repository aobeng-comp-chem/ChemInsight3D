# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:31:09 2026

@author: aoben
"""

import ast
import textwrap
from pathlib import Path


PYTHON_FUNCTIONS = r"""
{
   'd0': lambda dx, dy, dz: 0.5 * (2*dz*dz - dx*dx - dy*dy),
        'dc1': lambda dx, dy, dz: 1.7320508075688773 * dy * dz,
        'ds1': lambda dx, dy, dz: 1.7320508075688773 * dx * dz,
        'dc2': lambda dx, dy, dz: 0.86602540378443865 * (dx*dx - dy*dy),
        'ds2': lambda dx, dy, dz: 1.7320508075688773 * dx * dy,
}
"""




class PythonToCppExpression:
    """Convert a limited Python mathematical AST into C++ syntax."""

    precedence = {
        ast.Add: 10,
        ast.Sub: 10,
        ast.Mult: 20,
        ast.Div: 20,
    }

    def convert(self, node, parent_precedence=0):
        method_name = f"_convert_{type(node).__name__}"
        method = getattr(self, method_name, None)

        if method is None:
            raise TypeError(
                f"Unsupported Python syntax: {ast.dump(node)}"
            )

        result, node_precedence = method(node)

        if node_precedence < parent_precedence:
            return f"({result})"

        return result

    def _convert_Name(self, node):
        return node.id, 100

    def _convert_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            raise TypeError(
                f"Only numerical constants are supported: {node.value!r}"
            )

        return repr(node.value), 100

    def _convert_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            operand = self.convert(node.operand, 30)
            return f"-{operand}", 30

        if isinstance(node.op, ast.UAdd):
            operand = self.convert(node.operand, 30)
            return f"+{operand}", 30

        raise TypeError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )

    def _convert_BinOp(self, node):
        # Handle Python powers: x**n
        if isinstance(node.op, ast.Pow):
            return self._convert_power(node)

        operators = {
            ast.Add: " + ",
            ast.Sub: " - ",
            ast.Mult: " * ",
            ast.Div: " / ",
        }

        operator_type = type(node.op)

        if operator_type not in operators:
            raise TypeError(
                f"Unsupported binary operator: {operator_type.__name__}"
            )

        precedence = self.precedence[operator_type]
        operator = operators[operator_type]

        left = self.convert(node.left, precedence)

        # Parenthesize the right side of subtraction and division when needed.
        right_precedence = precedence
        if isinstance(node.op, (ast.Sub, ast.Div)):
            right_precedence += 1

        right = self.convert(node.right, right_precedence)

        return f"{left}{operator}{right}", precedence



    def _convert_power(self, node):
        if not isinstance(node.right, ast.Constant):
            raise TypeError("The exponent must be an integer constant.")
    
        exponent = node.right.value
    
        if not isinstance(exponent, int) or exponent < 0:
            raise TypeError(
                "Only non-negative integer powers are supported."
            )
    
        base = self.convert(node.left, 100)
    
        if exponent == 0:
            return "1.0", 100
    
        if exponent == 1:
            return base, 100
    
        # Use powi for every exponent greater than one.
        return f"powi({base},{exponent})", 100

def convert_dictionary_to_cpp(
    source,
    section_comment="// g",
    output_file=None,
):
    """Convert a dictionary of Python lambdas into C++ if statements."""

    syntax_tree = ast.parse(
        textwrap.dedent(source),
        mode="eval",
    )

    dictionary = syntax_tree.body

    if not isinstance(dictionary, ast.Dict):
        raise TypeError(
            "The input must be a Python dictionary of lambda functions."
        )

    converter = PythonToCppExpression()
    cpp_lines = [section_comment]

    for key_node, value_node in zip(
        dictionary.keys,
        dictionary.values,
    ):
        if not isinstance(key_node, ast.Constant):
            raise TypeError("Dictionary keys must be strings.")

        if not isinstance(value_node, ast.Lambda):
            raise TypeError(
                f"The value associated with {key_node.value!r} "
                "must be a lambda function."
            )

        orbital_name = key_node.value
        cpp_expression = converter.convert(value_node.body)

        cpp_lines.append(
            f'if (orb_val == "{orbital_name}") '
            f'return {cpp_expression};'
        )

    result = "\n".join(cpp_lines)

    if output_file is not None:
        Path(output_file).write_text(result + "\n", encoding="utf-8")

    return result


if __name__ == "__main__":
    cpp_code = convert_dictionary_to_cpp(
        PYTHON_FUNCTIONS,
        section_comment="// g",
        output_file="g_functions.cpp",
    )

    print(cpp_code)