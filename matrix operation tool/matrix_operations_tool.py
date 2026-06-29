"""
Matrix Operations Tool
=======================
An interactive command-line application for performing common matrix
operations using NumPy: addition, subtraction, multiplication, transpose,
and determinant calculation.

Run with: python matrix_operations_tool.py
"""

import numpy as np


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_matrix(matrix: np.ndarray, label: str = "Matrix") -> None:
    """Print a matrix in a clean, aligned, structured format."""
    print(f"\n{label} ({matrix.shape[0]}x{matrix.shape[1]}):")
    # Determine column width based on the longest formatted number
    formatted = [[format_number(v) for v in row] for row in matrix]
    col_width = max(len(v) for row in formatted for v in row) + 2

    border = "+" + "-" * (col_width * matrix.shape[1] + 1) + "+"
    print(border)
    for row in formatted:
        line = "".join(v.rjust(col_width) for v in row)
        print(f"|{line} |")
    print(border)


def format_number(value: float) -> str:
    """Format numbers cleanly: integers without decimals, floats rounded."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}"


# --------------------------------------------------------------------------
# Input helpers
# --------------------------------------------------------------------------

def get_dimensions(prompt: str) -> tuple:
    while True:
        try:
            rows = int(input(f"{prompt} - number of rows: ").strip())
            cols = int(input(f"{prompt} - number of columns: ").strip())
            if rows <= 0 or cols <= 0:
                print("  ! Rows and columns must be positive integers. Try again.")
                continue
            return rows, cols
        except ValueError:
            print("  ! Please enter valid integers.")


def input_matrix(label: str) -> np.ndarray:
    """Prompt the user to input a matrix row by row."""
    print(f"\n--- Enter values for {label} ---")
    rows, cols = get_dimensions(label)
    print(f"Enter each row as {cols} space-separated numbers (you'll enter {rows} rows).")

    data = []
    for r in range(rows):
        while True:
            raw = input(f"  Row {r + 1}: ").strip()
            values = raw.split()
            if len(values) != cols:
                print(f"  ! Expected {cols} values, got {len(values)}. Try again.")
                continue
            try:
                data.append([float(v) for v in values])
                break
            except ValueError:
                print("  ! All values must be numbers. Try again.")

    matrix = np.array(data)
    print_matrix(matrix, label=f"{label} (entered)")
    return matrix


def choose_existing_matrix(matrices: dict, prompt: str) -> np.ndarray:
    """Let the user pick a previously stored matrix or enter a new one."""
    if matrices:
        print(f"\n{prompt}")
        names = list(matrices.keys())
        for i, name in enumerate(names, 1):
            shape = matrices[name].shape
            print(f"  {i}. {name} ({shape[0]}x{shape[1]})")
        print(f"  {len(names) + 1}. Enter a new matrix")

        while True:
            choice = input("Choose an option: ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(names):
                    return matrices[names[idx - 1]]
                elif idx == len(names) + 1:
                    return input_matrix("New Matrix")
            print("  ! Invalid choice. Try again.")
    else:
        return input_matrix("New Matrix")


def store_matrix(matrices: dict, matrix: np.ndarray) -> None:
    name = input("Save this matrix as (name, or leave blank to skip): ").strip()
    if name:
        matrices[name] = matrix
        print(f"  ✓ Saved as '{name}'.")


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

def operation_addition(matrices: dict) -> None:
    print_header("MATRIX ADDITION")
    a = choose_existing_matrix(matrices, "Select first matrix (A):")
    b = choose_existing_matrix(matrices, "Select second matrix (B):")
    if a.shape != b.shape:
        print(f"\n  ✗ Error: shapes {a.shape} and {b.shape} are not compatible for addition.")
        return
    result = a + b
    print_matrix(a, "Matrix A")
    print_matrix(b, "Matrix B")
    print_matrix(result, "Result (A + B)")
    store_matrix(matrices, result)


def operation_subtraction(matrices: dict) -> None:
    print_header("MATRIX SUBTRACTION")
    a = choose_existing_matrix(matrices, "Select first matrix (A):")
    b = choose_existing_matrix(matrices, "Select second matrix (B):")
    if a.shape != b.shape:
        print(f"\n  ✗ Error: shapes {a.shape} and {b.shape} are not compatible for subtraction.")
        return
    result = a - b
    print_matrix(a, "Matrix A")
    print_matrix(b, "Matrix B")
    print_matrix(result, "Result (A - B)")
    store_matrix(matrices, result)


def operation_multiplication(matrices: dict) -> None:
    print_header("MATRIX MULTIPLICATION")
    a = choose_existing_matrix(matrices, "Select first matrix (A):")
    b = choose_existing_matrix(matrices, "Select second matrix (B):")
    if a.shape[1] != b.shape[0]:
        print(f"\n  ✗ Error: cannot multiply {a.shape} by {b.shape} "
              f"(columns of A must equal rows of B).")
        return
    result = a @ b
    print_matrix(a, "Matrix A")
    print_matrix(b, "Matrix B")
    print_matrix(result, "Result (A x B)")
    store_matrix(matrices, result)


def operation_transpose(matrices: dict) -> None:
    print_header("MATRIX TRANSPOSE")
    a = choose_existing_matrix(matrices, "Select matrix to transpose:")
    result = a.T
    print_matrix(a, "Original Matrix")
    print_matrix(result, "Transposed Matrix")
    store_matrix(matrices, result)


def operation_determinant(matrices: dict) -> None:
    print_header("DETERMINANT CALCULATION")
    a = choose_existing_matrix(matrices, "Select matrix:")
    if a.shape[0] != a.shape[1]:
        print(f"\n  ✗ Error: determinant requires a square matrix, got {a.shape}.")
        return
    det = np.linalg.det(a)
    print_matrix(a, "Matrix")
    print(f"\n  Determinant: {format_number(det)}")


def operation_view_saved(matrices: dict) -> None:
    print_header("SAVED MATRICES")
    if not matrices:
        print("  No matrices saved yet.")
        return
    for name, m in matrices.items():
        print_matrix(m, name)


# --------------------------------------------------------------------------
# Main menu
# --------------------------------------------------------------------------

MENU = """
Please choose an operation:
  1. Addition          (A + B)
  2. Subtraction        (A - B)
  3. Multiplication     (A x B)
  4. Transpose          (A^T)
  5. Determinant        (det(A))
  6. View saved matrices
  7. Exit
"""


def main() -> None:
    matrices = {}
    print_header("MATRIX OPERATIONS TOOL (NumPy)")
    print("Welcome! Enter matrices and perform operations interactively.")
    print("Tip: results can be saved and reused as inputs for later operations.")

    actions = {
        "1": operation_addition,
        "2": operation_subtraction,
        "3": operation_multiplication,
        "4": operation_transpose,
        "5": operation_determinant,
        "6": operation_view_saved,
    }

    while True:
        print(MENU)
        choice = input("Enter your choice (1-7): ").strip()

        if choice == "7":
            print("\nThank you for using the Matrix Operations Tool. Goodbye!")
            break
        elif choice in actions:
            try:
                actions[choice](matrices)
            except Exception as exc:  # safety net for unexpected input issues
                print(f"\n  ✗ Unexpected error: {exc}")
        else:
            print("  ! Invalid choice. Please enter a number from 1 to 7.")


if __name__ == "__main__":
    main()
