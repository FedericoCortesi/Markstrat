import os
import json
import numpy as np
from scipy.optimize import differential_evolution
from Utils import combined_error

class Solver:
    def __init__(self, attributes_path:str=None) -> None:

        if attributes_path is None:
            files = os.listdir("./Attributes")
            files.sort()
            file = files[-1]
            self.attributes_path = f"./Attributes/{file}"
        else:
            self.attributes_path = attributes_path

        print(f"Attributes file:{self.attributes_path}")

        with open(self.attributes_path, 'r') as file:
            self.attributes = json.load(file)
            
        # Obtain regression parameters
        self.semantic_params = self.attributes["DimSlopIntercept"]

        # Obtain MDS const
        self.mds_const = self.attributes["MDSConsts"]

        # Obtain MDS coeff
        self.mds_coef = self.attributes["SemMDSCoef"]



    def _name_to_code(self, name):
        feature_mapping = {
            "# Features": 1,
            "Design Index": 2,
            "Battery Life": 3,
            "Display Size": 4,
            "Proc. Power": 5,
            "Rec. retail price": 6
        }

        return feature_mapping[name]


    def regress_semantic(self, features_array):
        result = []
        # Iterate over features
        for n, feature in enumerate(features_array):
            
            # Find Values
            slope = self.semantic_params[str(n+1)]["Item1"]
            intercept = self.semantic_params[str(n+1)]["Item2"]

            # Regress
            sem_val = ((feature - intercept) / slope * 10) / 10
            result.append(sem_val)

        return result

    def regress_mds(self, features_array):
        result = []
        # iterate over mds
        for m in range(12, 14+1):
            intercept = self.mds_const[str(m)]
            mds_val = [intercept]
            # Iterate over features
            for n, feature in enumerate(features_array):
                # Find Values
                slope = self.mds_coef[str(n+1)][str(m)]

                # Regress
                sem_val = (feature*slope) 
                mds_val.append(sem_val)

            result.append(sum(mds_val))

        return result
    

    def find_optimum(self, ideal_semantic: list, ideal_mds: list, semantic_weights: list, mds_weights: list, error_weights: np.ndarray = None):
        # Define Feature bounds
        feature_bounds = [(5, 20), (3, 10), (24, 96), (4, 40), (5, 100), (215, 475)]

        # Run the optimization to find the best features that minimize the combined error
        result = differential_evolution(
            combined_error,
            bounds=feature_bounds,
            args=(ideal_semantic, ideal_mds, semantic_weights, mds_weights, error_weights, self)
        )

        optimal_features = result.x
        min_error = result.fun
        print("Optimal Features:", optimal_features)
        print("Minimum Combined Error:", min_error)

        return optimal_features, min_error


