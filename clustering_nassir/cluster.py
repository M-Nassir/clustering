import numpy as np
from perception_nassir import Perception

class NovelClustering:
    """
    A semi-supervised clustering algorithm based on anomaly detection.

    This method uses Nassir's anomaly detection method to iteratively refine clusters
    seeded with a tiny amount of labelled data. Anomalous points are ejected from 
    clusters and unlabelled points are tested for potential inclusion in clusters 
    over multiple refinement rounds.

    Note, data must be numeric and cluster labels must be integers >= 0; -1 reserved for anomalies.
    """

    def __init__(self, max_n_iterations=1000):
        """
        Initialise the clustering model.

        Parameters
        ----------
        max_n_iterations : int, optional
            Maximum number of refinement iterations to handle rare oscillations (default: 1000).
        """
        self.max_n_iterations = max_n_iterations  # in case of rare oscillations resulting in infinite loop
        self.clf_models = {}                      # Perception models for each cluster
        self.cluster_fit_scores_ = {}             # prediction scores of each cluster model on the entire dataset
        self.data_dimensionality_ = None          # dimensionality of the input data; used in predict stage
        self._valid_clusters = None               # list of valid cluster labels

        if not (isinstance(max_n_iterations, int) and max_n_iterations > 0):
            raise ValueError("max_n_iterations must be a positive integer")

    def _validate_input_array(self, X_with_label_column):
        """
        Validate the structure and content of the input array.

        Parameters
        ----------
        X_with_label_column : np.ndarray
            Input array with shape (n_samples, n_features + 1). 
            Last column must be integer labels representing the cluster labels with seeds and anomalies.

        Raises
        ------
        AssertionError
            If the input does not satisfy format or content constraints.
        """
        if not isinstance(X_with_label_column, np.ndarray):
            raise ValueError("Input must be a NumPy array.")
        if X_with_label_column.ndim != 2 or X_with_label_column.shape[1] < 2:
            raise ValueError("Input must be 2D with at least one feature column and one label column.")
        if np.isnan(X_with_label_column).any() or np.isinf(X_with_label_column).any():
            raise ValueError("Input contains NaN or infinite values.")
        
        X = X_with_label_column[:, :-1]
        labels = X_with_label_column[:, -1]

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Feature values must be numeric.")
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Label values must be numeric.")
        if not np.all(labels.astype(int) == labels):
            raise ValueError("All label values must be integers.")
        if not np.all(labels >= -1):
            raise ValueError("Label values must be ≥ -1 (-1 indicates unlabelled).")

        # Extract only the labelled points (i.e., those not marked as -1)
        # -1 reserved for anomalies label
        labelled_points = labels[labels != -1]

        # Count how many times each label appears
        unique_labels, label_counts = np.unique(labelled_points, return_counts=True)

        # Check for any labels with fewer than 3 points
        clusters_with_too_few_seeds = {}
        for label, count in zip(unique_labels, label_counts):
            if count < 3:
                clusters_with_too_few_seeds[int(label)] = int(count)

        if clusters_with_too_few_seeds:
            raise ValueError(
                "Each labelled cluster must have at least 3 seed samples. "
                f"Found too few for: {clusters_with_too_few_seeds}"
            )

    def _split_features_labels(self, X_with_label_column):
        """
        Split the input array into features and integer labels.

        Parameters
        ----------
        X_with_label_column : np.ndarray
            Input of shape (n_samples, n_features + 1).

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Integer labels containing seeds and anomalies.
        """
        self._validate_input_array(X_with_label_column)
        X = X_with_label_column[:, :-1]
        y = X_with_label_column[:, -1].astype(int)
        return X, y

    def _cluster_mse_scores(self, X, y_current_labels, cluster_labels):
        """
        Compute MSE scores for clusters and return clusters sorted by compactness.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_current_labels : np.ndarray
            Current labels including -1 for unlabelled.
        cluster_labels : np.ndarray
            Unique cluster labels.

        Returns
        -------
        sorted_clusters : np.ndarray
            Cluster labels sorted by increasing MSE.
        sorted_mse : np.ndarray
            Corresponding MSE values.
        """
        mse_array = np.full(len(cluster_labels), np.inf)
        label_to_index = {label: i for i, label in enumerate(cluster_labels)}

        for label in np.unique(y_current_labels):
            if label == -1:
                continue

            idx = label_to_index[label]
            cluster_points = X[y_current_labels == label]

            if cluster_points.shape[0] > 0:
                cluster_median = np.median(cluster_points, axis=0)
                mse = np.mean(np.square(cluster_points - cluster_median))
                mse_array[idx] = mse

        # Sort clusters by increasing MSE
        sort_indices = np.argsort(mse_array)
        sorted_clusters = cluster_labels[sort_indices]
        sorted_mse = mse_array[sort_indices]

        return sorted_clusters, sorted_mse

    def _fit_and_expand_clusters(self, X, y_current_labels):
        """
        Iteratively refine clusters based on seeded labels:
        - Fit Perception to each cluster
        - Eject anomalies
        - Loop to Re-fit if necessary
        - Try to claim nearby anomalies

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_current_labels : np.ndarray
            Current cluster assignments (including -1 for unlabelled/anomalous).

        Returns
        -------
        np.ndarray
            Updated labels after iterative refinement.
        """
        
        def fit_and_eject_anomalies(cluster_label):
            """Fit model on cluster and eject detected outliers. Re-fit if necessary."""
            cluster_indices = np.where(y_current_labels == cluster_label)[0]
            if cluster_indices.size < 3:
                return None, False  # Not enough points

            X_cluster = X[cluster_indices]
            clf = Perception()
            clf.fit_predict(X_cluster)  # 0 = inlier, 1 = anomaly
            is_anomaly = clf.labels_ == 1
            
            if np.any(is_anomaly):
                # eject anomalies and update the cluster labels
                y_current_labels[cluster_indices] = -1
                inlier_mask = clf.labels_ == 0
                y_current_labels[cluster_indices[inlier_mask]] = cluster_label

                # Refit current cluster model only on inliers (post-ejection)
                updated_indices = cluster_indices[inlier_mask]
                if len(updated_indices) >= 3:
                    clf.fit(X[updated_indices])

            return clf, np.any(is_anomaly)

        def claim_cluster_anomalies(cluster_label, clf):
            """Use fitted model to claim anomalies if they fit cluster."""

            if clf is None:
                return False
        
            anomaly_indices = np.where(y_current_labels == -1)[0]
            if len(anomaly_indices) == 0:
                return False

            X_anomalies = X[anomaly_indices]
            preds = clf.predict(X_anomalies)
            accepted = (preds == 0)

            if np.any(accepted):
                # Only update accepted anomaly labels
                y_current_labels[anomaly_indices[accepted]] = cluster_label
                return True
            return False

        # ** Main loop **
        for _ in range(self.max_n_iterations):
            cluster_changed = False

            for cluster_label in self._valid_clusters:
                clf, ejected = fit_and_eject_anomalies(cluster_label)
                claimed = claim_cluster_anomalies(cluster_label, clf) if clf is not None else False

                if ejected or claimed:
                    cluster_changed = True

            # exit loop if no clusters changed
            if cluster_changed is False:
                break

        return y_current_labels
      
    def _final_claim_anomalies(self, X, y_current_labels):
        """
        Final pass where clusters try to claim remaining anomalies.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y_current_labels : np.ndarray
            Current cluster labels (including -1 for anomalies).

        Returns
        -------
        np.ndarray
            Updated cluster labels after claiming anomalies.
        """
        anomalies_mask = (y_current_labels == -1)
        if not np.any(anomalies_mask):
            return y_current_labels  # No anomalies to process

        anomaly_idx = np.where(anomalies_mask)[0]
        anomaly_points = X[anomaly_idx]

        for cluster_label in self._valid_clusters:

            clf = self.clf_models.get(cluster_label, None)
            if clf is None:
                continue  # no fitted model for this cluster, skip

            preds = clf.predict(anomaly_points)
            accepted = (preds == 0)

            if np.any(accepted):
                y_current_labels[anomaly_idx[accepted]] = cluster_label
                # Update anomalies
                anomaly_idx = anomaly_idx[~accepted]
                anomaly_points = anomaly_points[~accepted]
                if anomaly_idx.size == 0:
                    break  # all anomalies claimed

        return y_current_labels

    def _fit_final_classifiers(self, X, y_current_labels):
        """
        Trains a Perception classifier for each valid cluster using
        the current labels and stores the fitted classifiers.
        """
        self.clf_models = {}
        self.cluster_fit_scores_ = {}

        for cluster_label in self._valid_clusters:
            indices = np.where(y_current_labels == cluster_label)[0]
            if len(indices) == 0:
                continue
            
            X_sub = X[indices]
            clf = Perception()
            clf.fit(X_sub)

            # Predict scores across all data
            clf.predict(X)
        
            self.clf_models[cluster_label] = clf
            self.cluster_fit_scores_[cluster_label] = clf.scores_

    def fit(self, X_with_label_column: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model to the provided semi-supervised dataset.

        Parameters
        ----------
        X_with_labels : np.ndarray
            A 2D NumPy array where the last column contains integer labels.
            Label `-1` denotes unlabelled (anomalous) points, others are initial cluster labels.

        Returns
        -------
        np.ndarray
            Final cluster assignments after iterative refinement.
        """
        # Validate and extract features and labels from the input array
        X, y_current_labels = self._split_features_labels(X_with_label_column)

        # Store feature dimensionality for later validation during prediction
        self.data_dimensionality_ = X.shape[1]

        # Cluster ordering by compactness (MSE from median)
        unique_cluster_labels = np.unique(y_current_labels)
        ordered_cluster_labels_, _ = self._cluster_mse_scores(X, y_current_labels, unique_cluster_labels)

        # Store valid cluster labels for building models
        self._valid_clusters = [label for label in ordered_cluster_labels_ if label != -1]

        # Compute the cluster models from seeds
        y_current_labels = self._fit_and_expand_clusters(X, y_current_labels)
        y_current_labels = self._final_claim_anomalies(X, y_current_labels)
        self._fit_final_classifiers(X, y_current_labels)
        
        return y_current_labels

    def predict(self, X):
        """
        Predict cluster assignments for new data points based on trained cluster models.

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



