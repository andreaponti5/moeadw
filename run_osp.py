import json
import pandas as pd

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from tqdm import tqdm

from algo.bitflip_mutation import BitflipMutation
from algo.moead import MOEAD
from algo.moeadw import MOEADW
from algo.res_callback import ResCallback
from algo.sensor_crossover import SensorCrossover

from problem.osp import OSP2, OSP4


def get_algorithm(algorithm_name, m, seed):
    ref_dirs = get_reference_directions("uniform", m, n_partitions=6, seed=seed)
    if algorithm_name == "moead":
        return MOEAD(ref_dirs,
                     sampling=IntegerRandomSampling(),
                     mutation=BitflipMutation(),
                     crossover=SensorCrossover(),
                     seed=seed)
    elif algorithm_name == "moeadw":
        return MOEADW(ref_dirs,
                      sampling=IntegerRandomSampling(),
                      mutation=BitflipMutation(),
                      crossover=SensorCrossover(),
                      seed=seed)
    return None


# Problem configuration
network = "Neptun"
budget = 10
nobj = 2

# Experiment configuration
ntrial = 10
ngen = 400

# Impact matrices for detection time and volume of contaminated water
impact_matrix1 = pd.read_csv(f"data/{network}_det_times.csv",
                             dtype={'sensor': 'str', 'scenario': 'str', 'time': 'float'})
impact_matrix2 = pd.read_csv(f"data/{network}_vol_contam.csv",
                             dtype={'sensor': 'str', 'scenario': 'str', 'volume': 'float'})

# Loop over different algorithms
for algo_name in ["moead", "moeadw"]:
    if nobj == 2:
        prob = OSP2(impact_matrix1, budget)
    elif nobj == 4:
        prob = OSP4(impact_matrix1, impact_matrix2, budget)
    else:
        raise "Not implemented yet! Only 2 and 4 objectives are supported!"

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
    res["config"]["budget"] = prob.budget
    res["config"]["ntrial"] = ntrial
    res["config"]["ngen"] = ngen
    res["config"]["test_function"] = network
    res["config"]["algo"] = algo_name

    # Store the results in a json
    json.dump(res, open(f"results/OSP{prob.n_obj}_{network}_{prob.budget}_{algo_name}.json", "w"))
