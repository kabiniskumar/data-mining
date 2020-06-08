import sys
import csv
from numpy import NaN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsfresh.feature_extraction.feature_calculators as fc
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy.polynomial.polynomial as poly
import pickle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def csv_parser(input_folder):
    # Adding NaN values to ensure that all tuples in the data files have the same number of attributes
    categories = ["Nomeal", "mealData"]
    for i in categories:
        lines = []
        for j in range(1, 6):
            file_name = input_folder + str(i) + str(j) + ".csv"
            with open(file_name, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for k, line in enumerate(reader):
                    n_cols = len(line)
                    if n_cols < 30:
                        delta = 30 - n_cols
                        line = line + [NaN] * delta
                    lines.append(line)

        output_file = i + ".csv"
        writer = csv.writer(open(output_file, "w+"))
        writer.writerows(lines)

    return

# Preprocessing the data
def preprocessing(filename):

    input_file = filename
    columns = [i  for i in range(30)]
    #time_stamp_df = pd.read_csv("./MealNoMealData/mealData4.csv", usecols=columns, header=None)
    time_stamp_df = pd.read_csv(input_file, usecols=columns, header=None)

    # Considering only 30 time instances, eliminating the rest
    time_stamp_df = time_stamp_df.iloc[:, :30]

    # Omitting rows that contain less than 21 non-null values
    filtered_ts_df = time_stamp_df.dropna(thresh=21)

    # Computing values for missing values
    missing_value_indices = []
    for i in range(len(filtered_ts_df)):
        row = filtered_ts_df.iloc[i]
        indices = []

        for i in range(len(row) - 1):
            if np.isnan(row[i]):
                indices.append(i)

        missing_value_indices.append(indices)

    non_null_indices = []
    for indices in missing_value_indices:
        if indices:
            lower_bound = indices[0] - 1
            upper_bound = indices[-1] + 1
            if upper_bound != 29 and lower_bound != -1:
                non_null_indices.append((lower_bound, upper_bound))
                continue
        non_null_indices.append(())


    for i in range(len(filtered_ts_df)):
        if non_null_indices[i]:
            lower_bound = non_null_indices[i][0]
            upper_bound = non_null_indices[i][1]
            if (upper_bound - lower_bound) == 2:
                null_index = int(np.mean((upper_bound, lower_bound)))
                imputed_value = (filtered_ts_df.iloc[i][upper_bound] + filtered_ts_df.iloc[i][lower_bound]) / 2
                filtered_ts_df.iloc[i][null_index] = round(imputed_value)
            else:
                lower_bound = non_null_indices[i][0]
                upper_bound = non_null_indices[i][1]

                glucose_difference = filtered_ts_df.iloc[i][lower_bound] - filtered_ts_df.iloc[i][upper_bound]
                number_of_increments = upper_bound - lower_bound
                k = filtered_ts_df.iloc[i][upper_bound]
                for j in range(number_of_increments - 1):
                    imputed_value = k + glucose_difference / number_of_increments
                    filtered_ts_df.iloc[i][lower_bound + j + 1] = round(imputed_value)
                    k = imputed_value

    # Replacing Null values at either end of a row with the corresponding row average
    filtered_ts_df = filtered_ts_df.apply(lambda row: row.fillna(round(row.mean())), axis=1)

    output_file = "preprocessed_" + input_file
    print(output_file)
    filtered_ts_df.to_csv(output_file, index=False)

    return

def feature_extraction():
    cgm_glucose_levels_no_meal = pd.read_csv("preprocessed_Nomeal.csv")
    cgm_glucose_levels_meal = pd.read_csv("preprocessed_mealData.csv")

    cgm_glucose = np.concatenate((cgm_glucose_levels_meal.values, cgm_glucose_levels_no_meal.values), axis=0)

    n = len(cgm_glucose[0])
    chunk_size = n//4

    rows = []

    for row in cgm_glucose:
        val =[]

        # Feature set 1 : Windowed Mean
        for i in range(0,30,6):
            val.append(fc.mean(row[i:i+6]))

        # Feature set 2: Windowed Variance
        # for i in range(0,30,6):
        #     val.append(fc.variance(row[i:i+6]))
        for i in range(0,30,6):
            val.append(fc.sum_of_reoccurring_data_points(row[i:i+6]))

        # Computing the top 5 fft coefficients
        fft_coefficients = fft(row, n=6)[1:]
        fft_coefficients_real = [value.real for value in fft_coefficients]
        val += fft_coefficients_real

        #val.append(np.sqrt(np.mean(row[24:]**2)))

        # Feature set 4: Polyfit

        x = np.linspace(0, 1, len(row))
        y = row
        val.extend(poly.polyfit(x, y, 3)[:-1])

        rows.append(val)

        # for i in range(0, 30, 6):
        #     val.append(fc.change_quantiles(row[i:i + 6],))


    feature_matrix = pd.DataFrame(StandardScaler().fit_transform(rows))
    labels = [1]*len(cgm_glucose_levels_meal)
    label_no = [0]*len(cgm_glucose_levels_no_meal)
    labels.extend(label_no)
    labels = np.array(labels)

    feature_matrix_meal = feature_matrix.iloc[:len(cgm_glucose_levels_meal),:].values
    feature_matrix_no_meal = feature_matrix.iloc[len(cgm_glucose_levels_meal):,:].values

    pca = PCA()
    pca.fit(feature_matrix_meal)
    # print("PCA explained variance: ")
    # print(pca.explained_variance_ratio_)

    # Saving new dim-space
    pd.DataFrame(pca.components_[:5]).to_csv("pca_components.csv", header=None, index=None)

    transformedData = np.dot(feature_matrix_meal, np.transpose(pca.components_[:5]))
    transformedData_no_meal = np.dot(feature_matrix_no_meal, np.transpose(pca.components_[:5]))

    transformedData = np.concatenate((transformedData,transformedData_no_meal))

    return transformedData, labels


def run_model(feature_matrix, labels):

    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)

    random_forest = RandomForestClassifier(max_depth=4, n_estimators=150)
    svm = SVC(gamma="auto")
    gradient_boosting = GradientBoostingClassifier(learning_rate=0.01, n_estimators=105)
    neural_network = MLPClassifier(alpha=0.0001, max_iter=400)

    classifiers = [random_forest, svm, gradient_boosting, neural_network]
    classification_algos = ["random_forest", "svm", "gradient_boosting", "neural_network"]

    for i in range(len(classifiers)):
        print("\n\nClassifier: " + classification_algos[i])
        print("Performing k-fold cross validation (k=5)")

        best_clf = None
        best_accuracy, best_precision, best_recall = 0, 0, 0
        for train_index, test_index in kf.split(feature_matrix):
            X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            clf = classifiers[i].fit(X_train, y_train)

            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions) * 100

            metrics = precision_recall_fscore_support(y_test, predictions, average='macro')
            precision, recall = metrics[0], metrics[1]
            f1_score = 2 * (precision * recall) / (precision + recall)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_clf = clf
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score

        print("Performance Metrics chosen from the k-fold iteration with best performance: ")
        print("Accuracy = " + str(best_accuracy) + "%")
        print("Precision = " + str(best_precision))
        print("Recall = " + str(best_recall))
        print("F1 Score = " + str(best_f1_score))
        model_filename = classification_algos[i] + "_model.sav"
        pickle.dump(best_clf, open(model_filename, 'wb'))


if __name__ == '__main__':
    # folder = sys.argv[1]
    folder = input("Enter the path name: ")
    csv_parser(folder)
    preprocessing("mealData.csv")
    preprocessing("Nomeal.csv")
    feature_matrix, labels = feature_extraction()
    run_model(feature_matrix, labels)
