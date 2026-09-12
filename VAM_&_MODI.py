import numpy as np

def vogel_approximation_method(cost, supply, demand):
    """Calculates Initial Basic Feasible Solution using VAM."""
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    cost_copy = cost.astype(float).copy()
    
    rows, cols = cost.shape
    allocation = np.zeros((rows, cols))
    
    while sum(supply_copy) > 0 and sum(demand_copy) > 0:
        row_penalties = []
        col_penalties = []
        
        # Calculate Row Penalties
        for i in range(rows):
            if supply_copy[i] > 0:
                valid_costs = [cost_copy[i][j] for j in range(cols) if demand_copy[j] > 0]
                if len(valid_costs) > 1:
                    sorted_costs = sorted(valid_costs)
                    row_penalties.append((sorted_costs[1] - sorted_costs[0], i))
                elif len(valid_costs) == 1:
                    row_penalties.append((valid_costs[0], i))
                else:
                    row_penalties.append((-1, i))
            else:
                row_penalties.append((-1, i))
                
        # Calculate Column Penalties
        for j in range(cols):
            if demand_copy[j] > 0:
                valid_costs = [cost_copy[i][j] for i in range(rows) if supply_copy[i] > 0]
                if len(valid_costs) > 1:
                    sorted_costs = sorted(valid_costs)
                    col_penalties.append((sorted_costs[1] - sorted_costs[0], j))
                elif len(valid_costs) == 1:
                    col_penalties.append((valid_costs[0], j))
                else:
                    col_penalties.append((-1, j))
            else:
                col_penalties.append((-1, j))

        max_row_penalty = max(row_penalties, key=lambda x: x[0])
        max_col_penalty = max(col_penalties, key=lambda x: x[0])

        if max_row_penalty[0] >= max_col_penalty[0]:
            r = max_row_penalty[1]
            valid_cols = [(cost_copy[r][j], j) for j in range(cols) if demand_copy[j] > 0]
            c = min(valid_cols, key=lambda x: x[0])[1]
        else:
            c = max_col_penalty[1]
            valid_rows = [(cost_copy[i][c], i) for i in range(rows) if supply_copy[i] > 0]
            r = min(valid_rows, key=lambda x: x[0])[1]

        # Allocate minimum of supply or demand
        allocated_qty = min(supply_copy[r], demand_copy[c])
        allocation[r][c] = allocated_qty
        
        supply_copy[r] -= allocated_qty
        demand_copy[c] -= allocated_qty

    return allocation

def solve_modi(cost, allocation):
    """Optimizes the solution using the MODI (Modified Distribution) Method."""
    rows, cols = cost.shape
    
    def get_basic_cells(alloc):
        return [(i, j) for i in range(rows) for j in range(cols) if alloc[i][j] > 0]

    def handle_degeneracy(alloc, basic_cells):
        """Adds zero-allocations (epsilon) to independent cells if basic cells < m + n - 1"""
        required_cells = rows + cols - 1
        while len(basic_cells) < required_cells:
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in basic_cells:
                        # Simple heuristic: add to first available empty cell
                        basic_cells.append((i, j))
                        alloc[i][j] = 1e-10  # Very small value representing epsilon
                        if len(basic_cells) == required_cells:
                            return

    def get_uv(basic_cells):
        u = {i: None for i in range(rows)}
        v = {j: None for j in range(cols)}
        u[0] = 0  # Set arbitrary starting point
        
        while None in u.values() or None in v.values():
            for r, c in basic_cells:
                if u[r] is not None and v[c] is None:
                    v[c] = cost[r][c] - u[r]
                elif v[c] is not None and u[r] is None:
                    u[r] = cost[r][c] - v[c]
        return u, v

    def find_loop(start_cell, basic_cells):
        def dfs(current_node, is_horizontal, path, visited):
            if len(path) > 3 and current_node == start_cell:
                return path
            
            r, c = current_node
            for nr, nc in basic_cells + [start_cell]:
                if (nr, nc) not in visited or (nr, nc) == start_cell:
                    if is_horizontal and nr == r and nc != c:
                        res = dfs((nr, nc), not is_horizontal, path + [(nr, nc)], visited | {(nr, nc)})
                        if res: return res
                    elif not is_horizontal and nc == c and nr != r:
                        res = dfs((nr, nc), not is_horizontal, path + [(nr, nc)], visited | {(nr, nc)})
                        if res: return res
            return None
            
        path = dfs(start_cell, True, [start_cell], {start_cell})
        if not path:
            path = dfs(start_cell, False, [start_cell], {start_cell})
        return path[:-1] # Remove the repeated start cell at the end

    iteration = 1
    while True:
        basic_cells = get_basic_cells(allocation)
        handle_degeneracy(allocation, basic_cells)
        
        u, v = get_uv(basic_cells)
        
        # Calculate Opportunity Costs (Delta_ij) for non-basic cells
        penalties = []
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in basic_cells:
                    delta = cost[i][j] - (u[i] + v[j])
                    if delta < 0:
                        penalties.append((delta, (i, j)))
                        
        if not penalties:
            print(f"Optimal solution reached at Iteration {iteration}.")
            break
            
        # Select most negative penalty (Entering Cell)
        penalties.sort(key=lambda x: x[0])
        entering_cell = penalties[0][1]
        
        # Find closed loop for reallocation
        loop = find_loop(entering_cell, basic_cells)
        
        # Determine stepping stone (+, -, +, -) minimum allocation
        minus_cells = loop[1::2]
        theta_val, exiting_cell = min((allocation[r][c], (r, c)) for r, c in minus_cells)
        
        # Reallocate units along the loop
        for idx, (r, c) in enumerate(loop):
            if idx % 2 == 0:
                allocation[r][c] += theta_val # Add to positive cells
            else:
                allocation[r][c] -= theta_val # Subtract from negative cells
                
        # Clean up floating point epsilon zeroes
        allocation[exiting_cell[0]][exiting_cell[1]] = 0.0 
        
        iteration += 1

    # Remove epsilons for final display
    allocation = np.where(allocation < 1e-9, 0, allocation)
    return allocation

if __name__ == "__main__":
    # Case Study Setup
    # C(i,j): Cost per unit from Plant i to Market j
    cost_matrix = np.array([
        [3, 1, 7, 4],  # P1 Costs
        [2, 6, 5, 9],  # P2 Costs
        [8, 3, 3, 2]   # P3 Costs
    ])
    
    supply = np.array([300, 400, 500])
    demand = np.array([250, 350, 400, 200])
    
    print("--- 1. Initial Basic Feasible Solution (VAM) ---")
    ibfs_allocation = vogel_approximation_method(cost_matrix, supply, demand)
    
    ibfs_cost = np.sum(ibfs_allocation * cost_matrix)
    print("VAM Allocation Matrix:")
    print(ibfs_allocation)
    print(f"Total Initial Cost: ${ibfs_cost:.2f}\n")
    
    print("--- 2. Optimization Phase (MODI Method) ---")
    optimal_allocation = solve_modi(cost_matrix, ibfs_allocation)
    
    optimal_cost = np.sum(optimal_allocation * cost_matrix)
    print("\nOptimal Allocation Matrix:")
    print(optimal_allocation)
    print(f"Minimum Total Transportation Cost: ${optimal_cost:.2f}")
    
    print("\n--- Final Shipment Plan Breakdown ---")
    rows, cols = optimal_allocation.shape
    for i in range(rows):
        for j in range(cols):
            if optimal_allocation[i][j] > 0:
                print(f"Ship {int(optimal_allocation[i][j])} units from Plant {i+1} to Market {j+1} (@ ${cost_matrix[i][j]}/unit)")