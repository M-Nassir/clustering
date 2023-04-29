#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:08:06 2022

@author: nassirmohammad
"""
import pytest
from models.cluster import Nassir_clustering
import numpy as np
import pandas as pd


# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Test fit() function
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def test_label_column_is_only_integers():
    pass


def test_with_non_numpy_array_input_fails():

    df = pd.DataFrame({'A': ['hello', 'vignan', 'geeks'],
                       'B': ['vignan', 'hello', 'hello'],
                       'C': [1, 2, 3]})

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(df)


def test_with_empty_array_fails():

    empty_array = np.array([])

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(empty_array)


def test_with_zero_array():

    zero_array = np.zeros([7, 8])

    order_dependent_cluster = Nassir_clustering()

    cluster_labels = order_dependent_cluster.fit(zero_array)

    assert (cluster_labels == np.zeros(zero_array.shape[0])).all()


def test_with_none_array():

    none_array = np.array(None)

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(none_array)


def test_with_nan_ndarray():

    nan_array = np.empty((3, 3,))
    nan_array[:] = np.nan

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(nan_array)


def test_with_only_one_point_in_array():

    single_value_array = np.zeros([1, 1])

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(single_value_array)


def test_with_not_enough_dimensions_of_data():

    single_value_no_label = np.ones([1])

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(single_value_no_label)


def test_with_not_enough_dimensions_of_data2():

    single_value_no_label2 = np.ones([1, 1])

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(single_value_no_label2)


def test_with_only_one_point_and_one_anomaly_label_in_array():

    single_value_label_and_anomaly_array = np.ones([1, 2])
    single_value_label_and_anomaly_array[0][1] = -1

    order_dependent_cluster = Nassir_clustering()
    cluster_labels = order_dependent_cluster.fit(
        single_value_label_and_anomaly_array)

    assert cluster_labels == -1


def test_with_only_one_group_of_labelled_anomaly_points():

    one_anomaly_grouping = np.random.random((10, 11))
    one_anomaly_grouping[:, -1] = -1

    order_dependent_cluster = Nassir_clustering()
    cluster_labels = order_dependent_cluster.fit(one_anomaly_grouping)

    assert (cluster_labels == one_anomaly_grouping[:, -1]).all()


def test_label_numbers_are_out_of_supported_range_greater_than_minus_2():

    data_holder = []

    # label number
    i = -2

    for mu, sig in [(0, 1), (50, 1), (100, 1)]:

        temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                    size=10000), axis=1)

        temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
        temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
        data_holder.append(temp_data_with_label)

        # increment label number
        i = i + 1

    # we have cluster numbers 0, 1, 2 (no assumed anomalies although there
    # some anomalies with respect to a cluster since they are gaussian
    # distribution)
    data_main = np.concatenate(data_holder)

    order_dependent_cluster = Nassir_clustering()

    with pytest.raises(Exception):
        order_dependent_cluster.fit(data_main)


def test_fully_labelled_data_with_only_one_cluster_no_anomalies():

    data_holder = []

    # label number
    i = 0

    for mu, sig in [(0, 1), (50, 1), (100, 1)]:

        temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                    size=10000), axis=1)

        temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
        temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
        data_holder.append(temp_data_with_label)

    # we have cluster numbers 0, 1, 2 (no assumed anomalies although there
    # some anomalies with respect to a cluster since they are gaussian
    # distribution)
    data_main = np.concatenate(data_holder)

    order_dependent_cluster = Nassir_clustering()
    cluster_labels = order_dependent_cluster.fit(data_main)

    # given gaussian data, anomalies will likely be found around each distribution
    assert (np.unique(cluster_labels) == i)


def test_fully_labelled_data_with_no_anomalies():

    data_holder = []

    # label number
    i = 0

    for mu, sig in [(0, 1), (50, 1), (100, 1)]:

        temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                    size=10000), axis=1)

        temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
        temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
        data_holder.append(temp_data_with_label)

        # increment label number
        i = i + 1

    # we have cluster numbers 0, 1, 2 (no assumed anomalies although there
    # some anomalies with respect to a cluster since they are gaussian
    # distribution)
    data_main = np.concatenate(data_holder)

    order_dependent_cluster = Nassir_clustering()
    cluster_labels = order_dependent_cluster.fit(data_main)

    # given gaussian data, anomalies will likely be found around each distribution
    assert (np.unique(cluster_labels) == np.array([-1.,  0.,  1.,  2.])).all()


def test_fully_labelled_data_with_anomalous_cluster():

    data_holder = []

    # label number
    i = -1

    for mu, sig in [(0, 1), (50, 1), (100, 1)]:

        temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                    size=10000), axis=1)

        temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
        temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
        data_holder.append(temp_data_with_label)

        # increment label number
        i = i + 1

    # we have cluster numbers 0, 1, 2 (no assumed anomalies although there
    # some anomalies with respect to a cluster since they are gaussian
    # distribution)
    data_main = np.concatenate(data_holder)

    order_dependent_cluster = Nassir_clustering()
    cluster_labels = order_dependent_cluster.fit(data_main)

    # given gaussian data, anomalies will likely be found around each distribution
    assert (np.unique(cluster_labels) == np.array([-1.,  0.,  1.])).all()

# %%
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
#                           Test predict() function
#
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


def setup_fit():

    # setup code
    data_holder = []

    # label number
    i = -1

    for mu, sig in [(0, 1), (50, 1), (100, 1)]:

        temp_data = np.expand_dims(np.random.normal(loc=mu, scale=sig,
                                                    size=10000), axis=1)

        temp_labels = np.expand_dims(i * np.ones(temp_data.shape[0]), axis=1)
        temp_data_with_label = np.append(temp_data, temp_labels, axis=1)
        data_holder.append(temp_data_with_label)

        # increment label number
        i = i + 1

    # we have cluster numbers 0, 1, 2 (no assumed anomalies although there
    # some anomalies with respect to a cluster since they are gaussian
    # distribution)
    return np.concatenate(data_holder)


def test_predict_with_non_numpy_array_input_fails():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    df = pd.DataFrame({'A': ['hello', 'vignan', 'geeks'],
                       'B': ['vignan', 'hello', 'hello'],
                       'C': [1, 2, 3]})

    with pytest.raises(Exception):
        cluster.predict(df)


def test_predict_with_empty_array_fails():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    empty_array = np.array([])

    with pytest.raises(Exception):
        cluster.predict(empty_array)


def test_predict_with_zero_array():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    zero_array = np.zeros([7, 2])

    with pytest.raises(Exception):
        cluster.predict(zero_array)


def test_predict_with_none_array():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    none_array = np.array(None)

    with pytest.raises(Exception):
        cluster.predict(none_array)


def test_predict_with_all_nan_array():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    nan_array = np.empty((3, 2,))
    nan_array[:] = np.nan

    with pytest.raises(Exception):
        cluster.fit(nan_array)


def test_predict_with_empty_array_with_positive_dimensions():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    empty_with_dim = np.array([[]])

    with pytest.raises(Exception):
        cluster.fit(empty_with_dim)


def test_with_only_one_point_in_array():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    single_value_array = np.zeros([1, 1])

    with pytest.raises(Exception):
        cluster.fit(single_value_array)


def test_with_incorrect_dimensions_of_data():

    X = setup_fit()
    cluster = Nassir_clustering()
    cluster.fit(X)

    multi_dim = np.empty([1, 3, 4, 5])

    with pytest.raises(Exception):
        cluster.fit(multi_dim)

# todo: test predict function
