import numpy as np

def hour_to_sinus(df):
    hours = df["hour"].values
    h_in_rad = hours/(24*2*np.pi)
    hours = np.sin(h_in_rad)

    return hours

def diff(dataset):
    dataset_diff = dataset
    dataset_diff[:, 1:, :] = dataset[:, 1:, :] - dataset[:, :-1, :]
    return dataset_diff

def inverse_diff(diff_dataset, orig_dataset):
    inv_diff_dataset = diff_dataset
    last_value = orig_dataset[:, -1, :]
    inv_diff_dataset = diff_dataset + last_value