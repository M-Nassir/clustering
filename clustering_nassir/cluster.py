import numpy as np
from perception_nassir import Perception

# TODO: finish predict method using scores with respect to clusters?
# TODO: maybe put check which clusters are tightest after every round? or will it be too slow
# without much improvement?
# TODO: a point can belong to multiple clusters

class NovelClustering:
    """
    A semi-supervised clustering algorithm based on anomaly detection.

    This method uses Nassir's anomaly detection method to iteratively refine clusters.
    Anomalous points are ejected from clusters and unlabelled points are tested 
    for potential inclusion in clusters over multiple refinement rounds.
    """

    def __init__(self, max_iterations=1000):
        self.max_n_iterations = max_iterations # in case of rare oscillations resulting in infinite loop
        self.clf_models = {}                   # Perception models for each cluster
        self.cluster_fit_scores_ = {}          # scores of each cluster on the entire dataset
        self.data_dimensionality_ = None       # dimensionality of the input data
        self.ordered_cluster_labels_ = None    # ordered cluster labels based on compactness
        self.mse_of_clusters_ = None           # MSE of clusters; used for ordering

    def _validate_input_array(self, input_array):
        assert isinstance(input_array, np.ndarray), "Input must be a NumPy array"
        assert input_array.ndim == 2, "Input must be 2D"
        assert input_array.shape[1] >= 2, "Must have features and label column"
        labels = input_array[:, -1]
        assert np.issubdtype(labels.dtype, np.number), "Label column must be numeric"
        assert np.all(labels.astype(int) == labels), "Labels must be integers"
        assert np.all(labels >= -1), "Labels must be ≥ -1"

    def _split_features_labels(self, input_array):
        self._validate_input_array(input_array)
        X = input_array[:, :-1]
        y = input_array[:, -1].astype(int)
        return X, y

    def _cluster_mse_scores(self, X, y_live, cluster_labels):
        mse_array = np.full(len(cluster_labels), np.inf)
        label_to_index = {label: i for i, label in enumerate(cluster_labels)}

        for label in np.unique(y_live):
            if label == -1:
                continue

            idx = label_to_index[label]
            cluster_points = X[y_live == label]

            if cluster_points.shape[0] > 0:
                cluster_median = np.median(cluster_points, axis=0)
                mse = np.mean(np.square(cluster_points - cluster_median))
                mse_array[idx] = mse

        # Sort clusters by increasing MSE
        sort_indices = np.argsort(mse_array)
        sorted_clusters = cluster_labels[sort_indices]
        sorted_mse = mse_array[sort_indices]

        return sorted_clusters, sorted_mse

    def _fit_and_expand_clusters(self, X, y_live):
        """
        Iteratively refine clusters based on seeded labels:
        - Fit Perception to each cluster
        - Eject anomalies
        - Re-fit if necessary
        - Try to claim nearby anomalies

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_live : np.ndarray
            Current cluster assignments (including -1 for unlabelled/anomalous).

        Returns
        -------
        np.ndarray
            Updated labels after iterative refinement.
        """
        def fit_and_eject_anomalies(cluster_label):
            """Fit model on cluster and eject detected outliers."""
            cluster_indices = np.where(y_live == cluster_label)[0]
            if len(cluster_indices) == 0:
                return None, False

            X_cluster = X[cluster_indices]
            clf = Perception()
            clf.fit_predict(X_cluster)  
            labels_pred = clf.labels_ # 0 = inlier, 1 = anomaly

            anomalies_found = np.any(labels_pred != 0)
            # Update y_live in place
            y_live[cluster_indices] = np.where(labels_pred == 0, cluster_label, -1)
            return clf, anomalies_found

        def claim_anomalies(cluster_label, clf):
            """Try to reassign anomalies to this cluster using fitted model."""
            anomaly_indices = np.where(y_live == -1)[0]
            if len(anomaly_indices) == 0 or clf is None:
                return False

            X_anomalies = X[anomaly_indices]
            labels_pred = clf.predict(X_anomalies)
            accepted = (labels_pred == 0)

            if np.any(accepted):
                y_live[anomaly_indices] = np.where(accepted, cluster_label, -1)
                return True
            return False
    
        itr = 0                 # Iteration counter
        cluster_changed = True  # Cluster change occurred flag

        while cluster_changed and itr < self.max_n_iterations:
            cluster_changed = False
            itr += 1

            for cluster_label in self.ordered_cluster_labels_:
                if cluster_label == -1:
                    continue # Skip anomalies cluster

                clf, changed = fit_and_eject_anomalies(cluster_label)
                if changed:
                    # points were ejected, so cluster has changed
                    cluster_changed = True 

                    # Refit only on inliers (post-ejection)
                    updated_indices = np.where(y_live == cluster_label)[0]
                    if len(updated_indices) > 0:
                        clf.fit(X[updated_indices])

                if claim_anomalies(cluster_label, clf):
                    cluster_changed = True # points were claimed, so cluster has changed

        return y_live
      
    def _final_claim_anomalies(self, X, y_live):
        """
        Final pass where clusters try to claim remaining anomalies.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_live : np.ndarray
            Current cluster labels (including -1 for anomalies).

        Returns
        -------
        np.ndarray
            Updated cluster labels after claiming anomalies.
        """
        anomalies_mask = (y_live == -1)
        if not np.any(anomalies_mask):
            return y_live  # No anomalies to process

        anomaly_points = X[anomalies_mask]

        for cluster_label in self.ordered_cluster_labels_:
            if cluster_label == -1:
                continue

            X_sub = X[y_live == cluster_label]
            if X_sub.shape[0] <= 2:
                continue

            clf = Perception()
            clf.fit(X_sub)
            clf.predict(anomaly_points)

            y_live[anomalies_mask] = np.where(
                clf.labels_ == 0,
                cluster_label,
                y_live[anomalies_mask]
            )

        return y_live

    def _fit_final_classifiers(self, X, y_live):
        """
        Fit a final Perception classifier for each cluster (excluding anomalies),
        and store both the model and its scores on the entire dataset.

        Parameters
        ----------
        X : np.ndarray
            The full feature matrix.
        y_live : np.ndarray
            The final cluster label assignments.
        """
        for cluster_label in self.ordered_cluster_labels_:
            if cluster_label == -1:
                continue  # Skip anomalies

            cluster_mask = (y_live == cluster_label)
            X_sub = X[cluster_mask]

            if X_sub.shape[0] == 0:
                continue  # Skip empty clusters

            clf_m = Perception()
            clf_m.fit(X_sub)
            self.clf_models[cluster_label] = clf_m
            
            clf_m.predict(X)
            self.cluster_fit_scores_[cluster_label] = clf_m.scores_


    def fit(self, input_array):
        """
        Fit the clustering model to the provided semi-supervised dataset.

        Parameters
        ----------
        input_array : np.ndarray
            A 2D NumPy array where the last column contains integer labels.
            Label `-1` denotes unlabelled (anomalous) points, others are initial cluster labels.

        Returns
        -------
        np.ndarray
            Final cluster assignments after iterative refinement.
        """
        # Validate and extract features and labels from the input array
        X, y_live = self._split_features_labels(input_array)

        # Store feature dimensionality for later validation during prediction
        self.data_dimensionality_ = X.shape[1]

        # Cluster ordering by compactness (MSE from median)
        unique_cluster_labels = np.unique(y_live)
        self.ordered_cluster_labels_, self.mse_of_clusters_ = self._cluster_mse_scores(X, y_live, unique_cluster_labels)

        y_live = self._fit_and_expand_clusters(X, y_live)
        y_live = self._final_claim_anomalies(X, y_live)
        self._fit_final_classifiers(X, y_live)
        return y_live

    def predict(self, X):
        """
        Predict cluster assignments for new data points based on trained Cluster models.

        For each point in `X`, the method computes its anomaly score under each 
        trained cluster model. The point is assigned to the cluster where it 
        receives the lowest score. If the lowest score is above zero, the point 
        is considered an anomaly and assigned label `-1`.

        Parameters
        ----------
        X : np.ndarray
            A 2D NumPy array of shape (n_samples, n_features) representing the input data.
            Must have the same number of features as the data used during `fit()`.

        Returns
        -------
        predicted_labels : np.ndarray
            A 1D array of length n_samples containing predicted cluster labels. 
            Labels are integers corresponding to cluster IDs, or -1 for anomalies.

        min_scores : np.ndarray
            A 1D array of length n_samples containing the minimum anomaly score
            for each point across all cluster models.

        Raises
        ------
        AssertionError
            If input is not a 2D NumPy array with the correct dimensionality.
        """
        assert isinstance(X, np.ndarray), "Input must be a NumPy array"
        assert X.ndim == 2, "Input must be 2D"
        assert X.shape[1] == self.data_dimensionality_, "Input dimension mismatch"

        cluster_ids = list(self.clf_models.keys())
        scores_matrix = np.stack([self.clf_models[cid].predict(X) for cid in cluster_ids], axis=1)

        min_score_indices = np.argmin(scores_matrix, axis=1)
        min_scores = scores_matrix[np.arange(X.shape[0]), min_score_indices]
        predicted_labels = np.array([cluster_ids[i] for i in min_score_indices])

        predicted_labels[min_scores > 0] = -1
        return predicted_labels, min_scores


