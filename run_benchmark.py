import json

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

from tqdm import tqdm

from algo.moeadw import MOEADW
from algo.res_callback import ResCallback


def get_algorithm(algorithm_name, m, seed):
    ref_dirs = get_reference_directions("uniform", m, n_partitions=5, seed=seed)
    if algorithm_name == "moead":
        return MOEAD(ref_dirs, seed=seed)
    elif algorithm_name == "moeadw":
        return MOEADW(ref_dirs, seed=seed)
    return None


# Problem configuration
test_function = "DTLZ2"
nvar = 50
nobj = 2

# Experiment configuration
ntrial = 10
ngen = 200

# Loop over different algorithms
for algo_name in ["moead", "moeadw"]:
    prob = get_problem(test_function.lower(), n_var=nvar, n_obj=nobj)
    termination = ('n_gen', ngen)
    res = {}

    # Loop over different trial
    for trial in tqdm(range(ntrial), desc=algo_name, position=0):
        algo = get_algorithm(algo_name, m=prob.n_obj, seed=trial)
        pymoo_res = minimize(prob, algo, termination, seed=trial, callback=ResCallback(), verbose=False)
        res[trial] = pymoo_res.algorithm.callback.data

    # Add configuration to the results
    res["config"] = {}
    res["config"]["nobj"] = prob.n_obj
    res["config"]["nvar"] = prob.n_var
    res["config"]["ntrial"] = ntrial
    res["config"]["ngen"] = ngen
    res["config"]["test_function"] = test_function
    res["config"]["algo"] = algo_name
    if prob.n_obj > 3:
        res["PF"] = prob.pareto_front(get_reference_directions("uniform", prob.n_obj, n_partitions=5, seed=0)).tolist()
    else:
        res["PF"] = prob.pareto_front().tolist()

    # Store the results in a json file
    json.dump(res, open(f"results/{test_function}_{prob.n_obj}_{prob.n_var}_{algo_name}.json", "w"))
