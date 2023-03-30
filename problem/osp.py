import numpy as np

from pymoo.core.problem import ElementwiseProblem


def mean_impact(enabled_sensors, impact_matrix):
    result = 0
    impact_distr = {}
    for scenario_index in range(impact_matrix.shape[0]):
        tmp_sum = 0
        for sensor_index in range(impact_matrix.shape[1]):
            # Take the impact of the pair scenario-sensor
            d_scenario_sensor = impact_matrix[scenario_index][sensor_index]
            x_scenario_sensor = enabled_sensors[scenario_index][sensor_index]
            # Sum of impact of the active sensors
            tmp_sum += d_scenario_sensor * x_scenario_sensor
        impact_distr[scenario_index] = tmp_sum
        # Consider that all the scenarios have the same probability
        result += tmp_sum / impact_matrix.shape[0]
    return result, impact_distr


def std_impact(sp, impact_matrix):
    result = 0
    for scenario_index in range(impact_matrix.shape[0]):
        sensors_impact = []
        for sensor_index in range(impact_matrix.shape[1]):
            # Take the impact of the pair scenario-sensor
            d_scenario_sensor = impact_matrix[scenario_index][sensor_index]
            # If this sensor is active in the solution, its detection time must be considered computing the std
            if sp[sensor_index] == 1:
                sensors_impact.append(d_scenario_sensor)
        # Consider that all the scenarios have the same probability
        tmp_var = np.std(sensors_impact)
        result += tmp_var / impact_matrix.shape[0]
    return result


def active_sensors(sp, impact_matrix):
    # Get active sensors from solution
    active_sensors_list = [index for index, sensor in enumerate(sp) if sensor == 1]
    sensors_with_detection = []
    for scenarios_index in range(impact_matrix.shape[0]):
        scenario_row = list(np.zeros(impact_matrix.shape[1], dtype=int))
        min_det_tmp = 100000
        min_sensor_index = 0
        # Check which sensor out of the active sensor should be active in this scenario
        for sensor_index in active_sensors_list:
            det_time_tmp = impact_matrix[scenarios_index][sensor_index]
            if (det_time_tmp >= 0) & (det_time_tmp <= min_det_tmp):
                min_det_tmp = det_time_tmp
                min_sensor_index = sensor_index
        scenario_row[min_sensor_index] = 1
        sensors_with_detection.append(scenario_row)
    return sensors_with_detection


class OSP2(ElementwiseProblem):

    def __init__(self, trace_matrix, impact_matrix, budget):
        self.trace_matrix = trace_matrix
        self.impact_matrix = impact_matrix
        self.budget = budget

        self.scenarios = list(self.trace_matrix.columns)[0:-2]
        self.sensors = list(self.trace_matrix['Node'].unique())
        n_var = len(self.sensors)
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, 1),
                         type_var=int)

    def _evaluate(self, solution, out, *args, **kwargs):
        active_sensors_matrix = active_sensors(solution, self.impact_matrix)
        obj1, sf = mean_impact(enabled_sensors=active_sensors_matrix, impact_matrix=self.impact_matrix)
        obj2 = std_impact(sp=solution, impact_matrix=self.impact_matrix)
        out['F'] = [obj1, obj2]
        out['SF'] = list(sf.values())
        out['G'] = [self.budget_constraint(solution)]

    def budget_constraint(self, sp):
        return sum(sp) - self.budget


class OSP4(ElementwiseProblem):

    def __init__(self, trace_matrix, impact_matrix1, impact_matrix2, budget):
        self.trace_matrix = trace_matrix
        self.impact_matrix1 = impact_matrix1
        self.impact_matrix2 = impact_matrix2
        self.budget = budget

        self.scenarios = list(self.trace_matrix.columns)[0:-2]
        self.sensors = list(self.trace_matrix['Node'].unique())
        n_var = len(self.sensors)
        super().__init__(n_var=n_var,
                         n_obj=4,
                         n_ieq_constr=1,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, 1),
                         type_var=int)

    def _evaluate(self, solution, out, *args, **kwargs):
        active_sensors_matrix1 = active_sensors(solution, self.impact_matrix1)
        active_sensors_matrix2 = active_sensors(solution, self.impact_matrix2)
        obj1, sf1 = mean_impact(enabled_sensors=active_sensors_matrix1, impact_matrix=self.impact_matrix1)
        obj2 = std_impact(sp=solution, impact_matrix=self.impact_matrix1)
        obj3, sf2 = mean_impact(enabled_sensors=active_sensors_matrix2, impact_matrix=self.impact_matrix2)
        obj4 = std_impact(sp=solution, impact_matrix=self.impact_matrix2)
        out['F'] = [obj1, obj2, obj3, obj4]
        out['SF'] = list(sf1.values())
        out['G'] = [self.budget_constraint(solution)]

    def budget_constraint(self, sp):
        return sum(sp) - self.budget
