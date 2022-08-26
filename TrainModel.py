import pandas as pd
import matplotlib.pyplot as plt
from supervised.automl import AutoML
import sklearn.metrics
from IPython.utils import io

if __name__ == '__main__':
    train = {}
    train_names = ["real", "synt"]

    train["real"] = pd.read_csv('/data/notebooks/cancer/data/FinalTests/Biomarkers+Base/EndometrialCombined.csv',sep=';',decimal=',')
    train["synt"] = pd.read_csv('/data/notebooks/cancer/data/FinalTests/Biomarkers+Base/EndometrialCombinedAll_synthetic.csv',sep=',',decimal='.')

    # To simplify work, univariate and multivariate models are included in the same file, but the configuration for multivariate models is commented
    #
    feature_cols = {
        #"AllData": ['Age', 'BMI', 'sNeurpilin', 'sTie-2', 'IL-8', 'Follistatin', 'Leptin', 'G-CSF'],
        #"BMI-Biomarkers": ['BMI', 'sNeurpilin', 'sTie-2', 'IL-8', 'Follistatin', 'Leptin', 'G-CSF'],
        #"Biomarkers": [ 'sNeurpilin', 'sTie-2', 'IL-8', 'Follistatin', 'Leptin', 'G-CSF'],
        #"BMI-BiomarkersNoLeptin": ['BMI', 'sNeurpilin', 'sTie-2', 'IL-8', 'Follistatin', 'G-CSF'],
        #"BiomarkersNoBMINoLeptin": ['sNeurpilin', 'sTie-2', 'IL-8', 'Follistatin', 'G-CSF'],
        #"Selected": [ 'Age', 'IL-8', 'Leptin', 'G-CSF' ],
        "sNeurpilin": ['sNeurpilin' ],
        "sTie-2": [ 'sTie-2' ],
        "IL-8": [ 'IL-8' ],
        "Leptin": [ 'Leptin' ],
        "Follistatin": [ 'Follistatin' ],
        "G-CSF": [ 'G-CSF' ],
        "BMI": [ 'BMI' ],
    }

    for selectedModelName in feature_cols:
        for selectedTrainData in train_names:
            model = AutoML(results_path="./"+selectedModelName+"_"+selectedTrainData+"_compete_auc",
                          mode="Compete",
                          #total_time_limit=1200,
                          total_time_limit=900,
                          golden_feature=100,
                          explain_level=2,
                          n_jobs=16,
                          eval_metric="auc",
                          validation_strategy={
                            "validation_type": "kfold",
                            "k_folds": 20,
                            "shuffle": True,
                            "stratify": True,
                            "random_seed": 587
                          })

            X_train = train[selectedTrainData].loc[:, feature_cols[selectedModelName]]
            y_train = train[selectedTrainData].Case

            model.fit(X_train, y_train)
    
    
