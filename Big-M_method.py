import numpy as np

def formulate_and_solve():
    print("--- Interactive Big-M Simplex Solver ---")
    
    opt_type = input("Maximize or Minimize? (max/min): ").strip().lower()
    c_input = input("Enter objective function coefficients separated by space (e.g., 3 5): ")
    c = np.array([float(x) for x in c_input.split()])
    num_vars = len(c)
    
    num_constraints = int(input("Enter the number of constraints: "))
    
    A = []
    b = []
    relations = []
    
    for i in range(num_constraints):
        print(f"\nConstraint {i+1}:")
        coeffs = input(f"Enter coefficients for constraint {i+1} separated by space: ")
        a_row = [float(x) for x in coeffs.split()]
        
        relation = input("Enter relation (<=, >=, =): ").strip()
        rhs = float(input("Enter right-hand side (RHS) value: "))
        
        # Standard simplex initialization requires non-negative RHS
        if rhs < 0:
            a_row = [-x for x in a_row]
            rhs = -rhs
            if relation == "<=": relation = ">="
            elif relation == ">=": relation = "<="
            
        A.append(a_row)
        b.append(rhs)
        relations.append(relation)

    # Calculate required variables for Standard Form
    num_slacks = sum(1 for r in relations if r == "<=")
    num_surplus = sum(1 for r in relations if r == ">=")
    num_artificials = sum(1 for r in relations if r in [">=", "="])
    
    total_vars = num_vars + num_slacks + num_surplus + num_artificials
    
    # Initialize Simplex Tableau: Size is (Constraints + 1) x (Total Variables + RHS)
    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    M = 1e7  # Big M Penalty
    
    # Setup Objective Row (Row 0)
    # Z - cX + MA = 0 => Row starts with -c for Maximize
    # Min Z is solved by maximizing -Z => W + cX + MA = 0 => Row starts with +c
    if opt_type == "max":
        tableau[0, :num_vars] = -c
    else:
        tableau[0, :num_vars] = c
        
    col_idx = num_vars
    slack_idx, surplus_idx, art_idx = 1, 1, 1
    
    standard_form_eqs = []
    basis = [-1] * num_constraints
    art_indices = []
    
    # Populate constraint rows and build standard form strings
    for i in range(num_constraints):
        tableau[i+1, :num_vars] = A[i]
        tableau[i+1, -1] = b[i]
        
        # Format mathematical representation
        eq_str = f"{' + '.join([f'{A[i][j]}*x{j+1}' for j in range(num_vars)])}"
        eq_str = eq_str.replace("+ -", "- ")
        
        if relations[i] == "<=":
            tableau[i+1, col_idx] = 1 # Slack
            basis[i] = col_idx
            standard_form_eqs.append(eq_str + f" + S{slack_idx} = {b[i]}")
            col_idx += 1
            slack_idx += 1
            
        elif relations[i] == ">=":
            tableau[i+1, col_idx] = -1 # Surplus
            standard_form_eqs.append(eq_str + f" - E{surplus_idx} + A{art_idx} = {b[i]}")
            col_idx += 1
            surplus_idx += 1
            
            tableau[i+1, col_idx] = 1 # Artificial
            tableau[0, col_idx] = M
            art_indices.append((i+1, col_idx))
            basis[i] = col_idx
            col_idx += 1
            art_idx += 1
            
        elif relations[i] == "=":
            tableau[i+1, col_idx] = 1 # Artificial
            standard_form_eqs.append(eq_str + f" + A{art_idx} = {b[i]}")
            tableau[0, col_idx] = M
            art_indices.append((i+1, col_idx))
            basis[i] = col_idx
            col_idx += 1
            art_idx += 1

    print("\n--- Standard Form Formulation ---")
    if opt_type == "max":
        print(f"Maximize Z = {' + '.join([f'{c[i]}*x{i+1}' for i in range(len(c))])} - M*(Sum of Artificials)")
    else:
        print(f"Minimize Z = {' + '.join([f'{c[i]}*x{i+1}' for i in range(len(c))])} + M*(Sum of Artificials)")
        
    print("Subject to:")
    for eq in standard_form_eqs:
        print(f"  {eq}")
    print("  All variables >= 0")

    # Step: Zero out 'M' values in the objective row corresponding to basic artificial variables
    for r, c_idx in art_indices:
        tableau[0, :] -= M * tableau[r, :]

    print("\n--- Computational Solution ---")
    
    # --------------------------------
    # Big-M Simplex Algorithm Loop
    # --------------------------------
    status = "Optimal Solution Found"
    max_iters = 1000
    tolerance = 1e-7
    
    for _ in range(max_iters):
        # 1. Optimality Condition: All values in Row 0 (except RHS) >= 0
        if np.all(tableau[0, :-1] >= -tolerance):
            break
            
        # 2. Entering Variable (Most negative in Row 0)
        pivot_col = np.argmin(tableau[0, :-1])
        
        # 3. Ratio Test for Leaving Variable
        ratios = []
        for i in range(1, num_constraints + 1):
            if tableau[i, pivot_col] > tolerance:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
                
        # If all ratios are infinite, problem is unbounded
        if all(r == np.inf for r in ratios):
            status = "Optimization Failed: Unbounded Solution."
            break
            
        pivot_row = np.argmin(ratios) + 1
        
        # 4. Perform Row Operations (Pivoting)
        pivot_val = tableau[pivot_row, pivot_col]
        basis[pivot_row - 1] = pivot_col  # Update basic variable index for this row
        
        tableau[pivot_row, :] /= pivot_val # Normalize Pivot Row
        
        # Eliminate entries in the pivot column for all other rows
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    else:
        status = "Optimization Failed: Maximum iterations reached (Did not converge)."

    # 5. Feasibility Check: Check if an artificial variable is still basic & non-zero
    art_cols = [c_idx for _, c_idx in art_indices]
    for i, b_var in enumerate(basis):
        if b_var in art_cols and tableau[i+1, -1] > tolerance:
            status = "Optimization Failed: Infeasible Solution (Artificial variable remains strictly positive)."
            break

    # --------------------------------
    # Print the Results
    # --------------------------------
    if "Failed" in status:
        print(f"Status: {status}")
    else:
        print(f"Status: {status}")
        # The RHS of Row 0 holds the Optimal Value. If Minimize, flip the sign back.
        optimal_z = tableau[0, -1] if opt_type == "max" else -tableau[0, -1]
        print(f"Optimal Objective Value (Z): {optimal_z:.4f}")
        
        # Extract Decision Variables
        solution = [0.0] * num_vars
        for i in range(num_constraints):
            if basis[i] < num_vars: # True if the basic variable is an original decision variable
                solution[basis[i]] = tableau[i+1, -1]
                
        for i in range(num_vars):
            print(f"Decision Variable x{i+1} = {solution[i]:.4f}")

if __name__ == "__main__":
    formulate_and_solve()