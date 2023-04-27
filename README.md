# clustering
This project implements the semi-supervised clustering algorithm. The associated publication can be found at: XXX

The clustering algorithm takes as input an array of examples (rows) with numerical feature values (columns). 
An additional column in the input must hold the sample of labels obtained externally. The labels must be integers
with the value -1 reserved for examples whose labels are unknown. It is recommended that at least 10 examples are 
labelled for a particular group. 

The core of the clustering algorithm is Nassir's anomaly detection algorithm (called the perception). This ejects 
and adds points to a cluster, starting with the seeds of labelled data from a group. By iterating over the 
samples of labels it effectively grows clusters until no changes occur.

The clustering algorithm works with one-dimensional and multi-dimensional data. 

It will return labels for every example. The integer value -1 is reserved for the unknown or anomalous class; any
example that is deemed not to belong to any of the known clusters is deemed anomalous. These can be inspected in
subsequent rounds of analysis and then clustering run again until the user is satisfied. 
