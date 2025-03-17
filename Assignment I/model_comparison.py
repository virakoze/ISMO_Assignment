import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import time
import numpy as np

def solve_instance_comparison(data, sheet_name):
    """
    Solve both MILP and LP relaxation for a given instance and return their results
    Args:
        data: ProductionData instance containing the problem data
        sheet_name: Name of the instance being solved
    """
    def create_and_solve_model(is_relaxed=False):
        T = data.num_periods + 1  # Add 1 for the dummy period 0
        
        model = gp.Model("ULSP")
        
        # Decision variables - use continuous for LP relaxation
        y = model.addVars(T, vtype=GRB.CONTINUOUS if is_relaxed else GRB.BINARY, name="y")
        x = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="x")
        S = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="S")
        
        # Objective function
        model.setObjective(
            gp.quicksum(
                data.production_cost[t] * x[t] + data.setup_cost[t] * y[t] + data.holding_cost[t] * S[t] 
                for t in range(T)
            ), 
            GRB.MINIMIZE
        )
        
        # Constraints
        model.addConstr(S[0] == 0, name="no initial inventory")
        
        for t in range(1, T):
            model.addConstr(S[t-1] + x[t] == data.demand_forecast[t] + S[t])
            model.addConstr(x[t] <= sum(data.demand_forecast[m] for m in range(t, T)) * y[t])
        
        # Solve and time the model
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        return {
            'objective': model.objVal if model.status == GRB.OPTIMAL else None,
            'solve_time': solve_time,
            'status': model.status
        }
    
    # Solve both models
    milp_results = create_and_solve_model(is_relaxed=False)
    lp_results = create_and_solve_model(is_relaxed=True)
    
    # Calculate gap if both solutions are optimal
    if milp_results['status'] == GRB.OPTIMAL and lp_results['status'] == GRB.OPTIMAL:
        gap = (milp_results['objective'] - lp_results['objective']) / milp_results['objective'] * 100
    else:
        gap = None
    
    return {
        'instance': sheet_name,
        'milp_objective': milp_results['objective'],
        'milp_time': milp_results['solve_time'],
        'lp_objective': lp_results['objective'],
        'lp_time': lp_results['solve_time'],
        'gap_percentage': gap
    }

def solve_all_instances(file_path, sheet_list):
    """
    Solve all instances and return results as a DataFrame
    Args:
        file_path: Path to the Excel file containing the instances
        sheet_list: List of sheet names to process
    """
    results_list = []
    for sheet_name in sheet_list:
        print(f"\nProcessing instance: {sheet_name}")
        data = read_production_data(file_path, sheet_name)
        results = solve_instance_comparison(data, sheet_name)
        results_list.append(results)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results_list)
    return results_df 