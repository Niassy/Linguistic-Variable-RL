import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class FuzzyCollision:
    def __init__(self):
        # Define fuzzy variables (inputs and output)
        self.speed = ctrl.Antecedent(np.arange(0, 31, 1), 'speed')  # Speed from 0 to 30
        self.dist_same_lane = ctrl.Antecedent(np.arange(0, 81, 1), 'dist_same_lane')  # Distance on same lane
        self.dist_left_lane = ctrl.Antecedent(np.arange(0, 81, 1), 'dist_left_lane')  # Distance on left lane
        self.dist_right_lane = ctrl.Antecedent(np.arange(0, 81, 1), 'dist_right_lane')  # Distance on right lane
        self.action = ctrl.Consequent(np.arange(0, 5, 0.1), 'action')  # Action output (from 0 to 4)

        self._setup_membership_functions()
        self._define_rules()
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

        # Ensure the action output is linked properly
        self.simulation.output['action'] = 0  # Initialize the action output to avoid the KeyError

    def _setup_membership_functions(self):
        # Define fuzzy membership functions for speed (0-30)
        self.speed['very_slow'] = fuzz.trimf(self.speed.universe, [0, 0, 10])
        self.speed['slow'] = fuzz.trimf(self.speed.universe, [0, 10, 20])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [10, 15, 20])
        self.speed['fast'] = fuzz.trimf(self.speed.universe, [15, 25, 30])

        # Define fuzzy membership functions for distances (0-80)
        for antecedent in [self.dist_same_lane, self.dist_left_lane, self.dist_right_lane]:
            antecedent['very_short'] = fuzz.trimf(antecedent.universe, [0, 0, 10])
            antecedent['short'] = fuzz.trimf(antecedent.universe, [10, 20, 30])
            antecedent['medium'] = fuzz.trimf(antecedent.universe, [20, 35, 50])
            antecedent['long'] = fuzz.trimf(antecedent.universe, [40, 60, 80])

        # Define fuzzy membership functions for the output (action)
        self.action['LANE_LEFT'] = fuzz.trimf(self.action.universe, [0, 0, 1])
        self.action['IDLE'] = fuzz.trimf(self.action.universe, [1, 1, 2])
        self.action['LANE_RIGHT'] = fuzz.trimf(self.action.universe, [2, 2, 3])
        self.action['FASTER'] = fuzz.trimf(self.action.universe, [3, 3, 4])
        self.action['SLOWER'] = fuzz.trimf(self.action.universe, [4, 4, 5])

    def _define_rules(self):
        # Define fuzzy rules (collision avoidance rules included)
        rules = [
            # Basic driving rules based on speed and distance
            ctrl.Rule(self.speed['very_slow'] & self.dist_same_lane['very_short'], self.action['SLOWER']),
            ctrl.Rule(self.speed['slow'] & self.dist_same_lane['short'], self.action['IDLE']),
            ctrl.Rule(self.speed['medium'] & self.dist_same_lane['medium'], self.action['LANE_LEFT']),
            ctrl.Rule(self.speed['fast'] & self.dist_same_lane['long'], self.action['FASTER']),

            # Lane change and collision avoidance rules
            ctrl.Rule(self.dist_same_lane['very_short'], self.action['SLOWER']),
            ctrl.Rule(self.dist_same_lane['short'], self.action['IDLE']),
            ctrl.Rule(self.dist_left_lane['very_short'], self.action['LANE_LEFT']),
            ctrl.Rule(self.dist_right_lane['very_short'], self.action['LANE_RIGHT']),
            ctrl.Rule(self.dist_left_lane['short'], self.action['LANE_LEFT']),
            ctrl.Rule(self.dist_right_lane['short'], self.action['LANE_RIGHT']),
            ctrl.Rule(self.dist_left_lane['long'], self.action['IDLE']),
            ctrl.Rule(self.dist_right_lane['long'], self.action['IDLE']),
        ]
        self.rules = rules

    def compute_action(self, inputs):
        # Set inputs to the system
        for var, value in inputs.items():
            self.simulation.input[var] = value

        # Perform computation
        self.simulation.compute()

        # Check if the 'action' exists in the simulation output
        if 'action' not in self.simulation.output:
            raise KeyError("The output 'action' is missing from the simulation.")

        # Extract the crisp value and interpret it
        crisp_value = self.simulation.output['action']
        if crisp_value < 1:
            return 'LANE_LEFT'
        elif crisp_value < 2:
            return 'IDLE'
        elif crisp_value < 3:
            return 'LANE_RIGHT'
        elif crisp_value < 4:
            return 'FASTER'
        else:
            return 'SLOWER'

    def visualize(self, inputs):
        # Visualize inputs and outputs
        for var, value in inputs.items():
            getattr(self, var).view()
            plt.axvline(x=value, color='r', linestyle='--', label=f'Input: {value}')
            plt.title(var)
            plt.legend()
            plt.show()

        # Visualize the decision
        self.action.view(sim=self.simulation)
        plt.axvline(x=self.simulation.output['action'], color='r', linestyle='--', label='Decision')
        plt.title('Action')
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    system = FuzzyCollision()
    inputs = {
        'speed': 20,
        'dist_same_lane': 10,  # Distance to the closest vehicle in the same lane
        'dist_left_lane': 40,  # Distance to the closest vehicle in the left lane
        'dist_right_lane': 30,  # Distance to the closest vehicle in the right lane
    }
    action = system.compute_action(inputs)
    print(f"Decision: {action}")
    system.visualize(inputs)
