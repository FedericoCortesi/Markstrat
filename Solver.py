import os
import json
import numpy as np
import optuna
from scipy.optimize import differential_evolution
from Utils import combined_error, combined_error_minimum_distance

class Solver:
    def __init__(self, attributes_path:str=None, sector:str=None) -> None:

        if attributes_path is None:
            files = os.listdir(f"./Attributes/{sector}")
            files.sort()
            file = files[-1]
            self.attributes_path = f"./Attributes/{sector}/{file}"
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

        # Define features indexes
        self.sector = sector
        if self.sector == "Vodites":
            self.features_index = [7, 8, 9, 10, 11, 6]
            self.mds_index = [12, 15, 16]

        elif sector == "Sonites":
            self.features_index = [1, 2, 3, 4, 5, 6]
            self.mds_index = [12, 13, 14]
        else:
            raise ValueError("Non identified sector")



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
            index = str(self.features_index[n])

            # Find Values
            slope = self.semantic_params[index]["Item1"]
            intercept = self.semantic_params[index]["Item2"]

            if intercept == 0 or slope == 0:
                #print("Zero encountered, possible missing Data!")
                sem_val = 1

            else:
                # Regress
                sem_val = ((feature - intercept) / slope * 10) / 10
            
            result.append(sem_val)


        return np.array(result)

    def regress_mds(self, features_array):
        result = []
        # iterate over mds
        for m in range(3):
            mds_index = str(self.mds_index[m])

            intercept = self.mds_const[mds_index]
            mds_val = [intercept]

            if mds_val == 0:
                print("Zero encountered, possible missing Data!")
            
            # Iterate over features
            for n, feature in enumerate(features_array):
                feature_index = str(self.features_index[n])

                # Find Values
                slope = self.mds_coef[feature_index][mds_index]
                
                if slope == 0:
                    #print("Zero encountered, possible missing Data!")
                    pass

                # Regress
                sem_val = (feature*slope) 
                mds_val.append(sem_val)


            result.append(sum(mds_val))

        return np.array(result)
    

    def find_optimum(self, sector:str, ideal_semantic: list, ideal_mds: list, semantic_weights: list, mds_weights: list=[1/3, 1/3, 1/3], 
                     error_weights: np.ndarray = np.array([1,1])):
        if sector == "Sonites":
            # Define Feature bounds
            feature_bounds = [(5, 20), (3, 10), (24, 96), (4, 40), (5, 100), (215, 475)]
        
        elif sector == "Vodites":
            # Define Feature bounds
            feature_bounds = [(20, 100), (10, 100), (5, 50), (3, 10), (5, 100), (600, 800)]

        else:
            raise ValueError(f"Sector {sector} not recognized")


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



    def find_optimum_constrained(self, sector:str, ideal_semantic: list, ideal_mds: list, semantic_weights: list,
                                mds_weights: list = [1/3, 1/3, 1/3], error_weights: np.ndarray = np.array([1, 1])):
        if sector == "Sonites":
            # Define Feature bounds
            feature_bounds = [(5, 20), (3, 10), (24, 96), (4, 40), (5, 100), (215, 475)]
        
        elif sector == "Vodites":
            # Define Feature bounds
            feature_bounds = [(20, 100), (10, 100), (5, 50), (3, 10), (5, 100), (600, 800)]

        else:
            raise ValueError(f"Sector {sector} not recognized")

        def objective(trial):
            # Sample features within the defined bounds
            features = [trial.suggest_float(f'feature_{i}', bound[0], bound[1]) for i, bound in enumerate(feature_bounds)]
            
            # Calculate the combined error using the original function (assumed to be defined outside this context)
            combined_error = combined_error_minimum_distance(
                features,
                ideal_semantic, 
                ideal_mds, 
                semantic_weights, 
                mds_weights, 
                error_weights, 
                self
            )
            return combined_error

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study()
        study.optimize(objective, n_trials=1000)  # You can adjust the number of trials as needed

        optimal_features = study.best_params.values()
        min_error = study.best_value
        print("Optimal Features:", optimal_features)
        print("Minimum Combined Error:", min_error)

        return optimal_features, min_error
