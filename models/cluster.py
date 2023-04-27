import numpy as np
import pandas as pd
from perception import Perception

# TODO: fit/predict documentation
# TODO: predict method using scores


class Nassir_clustering:
    """Clustering algorithm.

    Semi-supervised clustering algorithm that uses Nassir's anomaly detection 
    algorithm at its core to eject or claim anomalies to grow clusters. 
    """

    def __init__(self):
        """Initialise Constructor for Nassir's clustering."""

    def fit(self, input_array):
        """Fit to data in order to cluster."""
        # ------------------------------------------------------------------
        # carry out iput data checks and setup
        # ------------------------------------------------------------------

        assert type(input_array) == np.ndarray, \
            "input_array must by a numpy ndarray"

        assert (input_array.ndim and input_array.size) > 0, \
            "input_array must not be empty or contain only 'None' "

        assert input_array.shape[1] >= 2,\
            "input array must have at least one feature, and one label column"

        assert np.all(np.mod(input_array[:, -1], 1) == 0), \
            "all input array labels (last column) must be integers"

        # copy the input arrays; last column is semi-supervised labels
        X = input_array[:, :-1].copy()
        y = input_array[:, -1].astype(int).copy()

        # extra checks, probably redundant
        assert (X.size and X.ndim) > 0, "data must be non-empty"
        assert (y.size and y.ndim) > 0, "cluster labels must be non-empty"

        # check label numbers are >= -1
        assert (y >= -1).all(), "All label values must be >= -1"

        # store the data dimensionality for cluster prediction stage
        self.data_dimensionality_ = X.shape[1]

        # store the perception classifier models
        self.clf_models = {}

        # dictionary to hold scores of every observation w.r.t. each cluster
        self.cluster_fit_scores_ = {}

        # get and sort the unique cluster labels
        self.cluster_numbers_ = np.sort(np.unique(y))

        # ------------------------------------------------------------------
        # sort cluster number order by minimum sum of squared error
        # ------------------------------------------------------------------

        # for each cluster compute the MSE from the median. Clusters will be
        # able to claim anomalies beginning with the tightest clusters.
        cluster_mse = []
        for cn in self.cluster_numbers_:
            if cn == -1:
                mse = -np.inf
            else:
                cluster_data = X[np.where(y == cn)]
                cluster_median = np.median(cluster_data, axis=0).reshape(1, -1)

                mse = np.square(cluster_data - cluster_median).mean()

            cluster_mse.append(mse)

        self.mse_of_clusters_ = np.array(cluster_mse)

        # sort the cluster order by MSE ascending
        self.cluster_numbers_ = self.cluster_numbers_[
            self.mse_of_clusters_.argsort()]

        # ------------------------------------------------------------------
        # cluster the data
        # ------------------------------------------------------------------

        # set counter for tracking validation
        cnt = 0

        # do loop until no point assignment changes occur
        cco = True
        while cco is True:

            # testing tracking; remove in future
            cnt = cnt + 1
            print(cnt)

            # this will be reset to true if a cluster assignment change occurs
            cco = False

            for i in self.cluster_numbers_:

                # if the index is for the anomaly label then skip
                # don't do clustering on anomaly group
                if i == -1:
                    continue

                # get the cluster data and indices
                X_sub = X[y == i]
                ind = np.where(y == i)

                # fit the classifier to the cluster data only
                clf = Perception()
                clf.fit_predict(X_sub)

                # anomalies -> -2 (eject anomalies and put into temp. category)
                y[ind] = np.where(clf.labels_ == 0, i, -2)

                # if anomalies have been ejected, modify the flag
                if np.any(clf.labels_ == -2):
                    cco = True

                    # re-fit clf to group without anomalies
                    clf.fit(X[y == i])

                # expand cluster only if it has >= 2 members
                if (X[y == i].shape[0] >= 2) and \
                        ((X[y == i].ndim and X[y == i].size) > 0):

                    # check anomalies for group membership
                    # -------------------------------------
                    anomaly_points = X[y == -1]

                    # check anomaly group is non-empty
                    if (anomaly_points.size and anomaly_points.ndim) > 0:

                        # get anomaly indices
                        ind0 = np.where(y == -1)

                        # predict anomalies w.r.t. the current group
                        clf.predict(anomaly_points)

                        # modify labels of any points belonging to group
                        y[ind0] = np.where(clf.labels_ == 0, i, -1)

                        # if anomalies predicted to belong to group
                        if np.any(clf.labels_ == 0):
                            cco = True

        # once initial clustering has finished, let each cluster consider each
        # point that has -2 label only once more.
        for i in self.cluster_numbers_:

            if i == -1:
                continue

            # get the cluster data points only
            X_sub = X[y == i]

            # get all the remaining anomalous points
            apl = X[y == -2]
            ind_anomalous_left = np.where(y == -2)

            # check if there are any left over anomalies
            if (apl.size and apl.ndim) <= 0:
                break

            # check if group is non-empty and has more than 2 elements
            if (X_sub.shape[0] >= 2) and (X_sub.size and X_sub.ndim) > 0:

                clf = Perception()
                clf.fit(X_sub)

                clf.predict(apl)

                # change label if it belongs to cluster, otherwise leave
                y[ind_anomalous_left] = np.where(clf.labels_ == 0, i, -2)

        # for temp anomalies not claimed by any cluster, convert to -1 label
        y[y == -2] = -1

        # finally let each cluster have chance to claim anomalies in case
        # they have changed after temp anomaly assignment changes.
        for i in self.cluster_numbers_:

            if i == -1:
                continue

            # get the cluster data points only
            X_sub = X[y == i]

            # get all the remaining anomalous points
            apl = X[y == -1]
            ind_anomalous_left = np.where(y == -1)

            # check if there are any left over anomalies
            if (apl.size and apl.ndim) <= 0:
                break

            # check if group is non-empty and has more than 2 elements
            if (X_sub.shape[0] >= 2) and (X_sub.size and X_sub.ndim) > 0:

                clf = Perception()
                clf.fit(X_sub)

                clf.predict(apl)

                # change label if it belongs to cluster, otherwise leave
                y[ind_anomalous_left] = np.where(clf.labels_ == 0, i, -1)

        # ------------------------------------------------------------------
        # save classifiers and get the scores by each cluster for every point
        # ------------------------------------------------------------------

        # calculate the centroids (mean or median) of each cluster. Note that
        # anomalies will not form part of the grouping.
        for i in self.cluster_numbers_:

            if i != -1:
                clf_m = Perception()
                clf_m.fit(X[y == i])

                # print(clf_m.multi_d_medians_.sum())

                # save the classifier model for the cluster
                self.clf_models[i] = clf_m

                clf_m.predict(X)

                # save the cluster scores for each model
                self.cluster_fit_scores_[i] = clf_m.scores_

        # return only the cluster labels for each example
        return y

    def predict(self, X):

        # TODO: prediction needs to be done by scores with respect to the
        # cluster, but it should be done by ...
        """ Predict cluster labels and associated score over new data.
        Input
        -------
        Array of values of same dimension as data upon which classifier has
        been fitted to.

        Returns
        -------
        Array of cluster numbers/anomaly labels for each observation.
        Array of cluster association score/anomaly score for each observation

        """
        assert type(X) == np.ndarray, "input_array must by a numpy array"

        assert X.shape[1] == self.data_dimensionality_, \
            "input array must match dimensionality of data fit stage"

        assert (X.size and X.ndim) > 0, "data must be non-empty"

        # fit and predict
        self.predict_model_scores_ = {}

        for cluster_number, clf_model in self.clf_models.items():
            clf_model.predict(X)
            self.predict_model_scores_[cluster_number] = clf_model.scores_

        df = pd.DataFrame(self.predict_model_scores_)

        # get the column number with minimum
        self.min_value_cluster_labels_ = df.idxmin(axis=1).values
        self.min_predict_score_ = df.min(axis=1).values

        # replace all positive scores with
        self.min_value_cluster_labels_[
            np.where(self.min_predict_score_ > 0)] = -1

        # return the dynamic cluster labels
        return self.min_value_cluster_labels_
