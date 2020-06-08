import pandas as pd
import numpy as np
import csv
from numpy import NaN
import tsfresh.feature_extraction.feature_calculators as fc
import numpy.polynomial.polynomial as poly
from math import pi
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def csv_parser():
    # Adding NaN values to ensure that all tuples in the data files have the same number of attributes

    file_name = input("Enter the file path: ")
    lines = []

    print(file_name)
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for k, line in enumerate(reader):
            n_cols = len(line)
            if n_cols < 30:
                delta = 30 - n_cols
                line = line + [NaN] * delta
            lines.append(line)

    output_file = "csv_parsed.csv"
    writer = csv.writer(open(output_file, "w+"))
    writer.writerows(lines)

    return

def preprocess():

    columns = [i  for i in range(30)]
    time_stamp_df = pd.read_csv("csv_parsed.csv", usecols=columns, header=None)

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

    feature_extraction(filtered_ts_df.values)

    return

def feature_extraction(cgm_glucose):

    n = len(cgm_glucose[0])
    chunk_size = n // 4

    rows = []

    for row in cgm_glucose:
        val = []

        # Feature set 1 : Mean
        for i in range(0, 30, 6):
            val.append(fc.mean(row[i:i + 6]))

        # Feature set 2: Sum of reoccurring data points
        for i in range(0, 30, 6):
            val.append(fc.sum_of_reoccurring_data_points(row[i:i + 6]))

        # Computing the top 5 fft coefficients
        fft_coefficients = fft(row, n=6)[1:]
        fft_coefficients_real = [value.real for value in fft_coefficients]
        val += fft_coefficients_real
        # val.extend(fft_coefficients_real)

        # Feature set 4: Polyfit
        x = np.linspace(0, 1, len(row))
        y = row
        val.extend(poly.polyfit(x, y, 3)[:-1])

        rows.append(val)

    feature_matrix = pd.DataFrame(StandardScaler().fit_transform(rows))
        # Transform data to training dimensional space

    pca_components = pd.read_csv("pca_components.csv", header=None)
    transformedData = np.dot(feature_matrix, np.transpose(pca_components))

        # Run all 4 classfication algorithms on test data

    classify_test_data(transformedData)



def classify_test_data(X_test):

    Y_test = [1]*len(X_test)

    classification_algos = ["random_forest", "svm", "gradient_boosting", "neural_network"]
    for i in range(len(classification_algos)):
        clf = pickle.load(open(classification_algos[i] + "_model.sav", "rb"))
        predictions = clf.predict(X_test)
        print("Prediction using "+ classification_algos[i])
        print(predictions)
        accuracy = accuracy_score(Y_test, predictions) * 100
        print("Accuracy: ",accuracy)


csv_parser()
preprocess()