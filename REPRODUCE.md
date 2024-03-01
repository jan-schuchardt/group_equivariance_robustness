# Reproducing our experiments

In the following, we describe which scripts and notebooks to run in which order to reproduce the different figures from our paper.

Note that you may have to adjust the directories in the individual config files (to point at the correct dataset folders, result folders etc.).  
You have to manually create these directories, they are not automatically created by the program itself.  
You will also need to adjust the slurm configuration parameters at the top of each file to match your cluster configuration (partition names etc.).

If you do not want to train and certify the models yourself, you can just run the plotting notebooks while keeping the flag `overwrite=False`.  
If you do, you will need to set `overwrite=True` when running the notebook for the first time.

## Point Clouds 

### Certificates (Fig. 3 & 7)
```
seml pointcloud_train add seml/configs/point_clouds/train/train_dgcnn.yaml start
seml pointcloud_train add seml/configs/point_clouds/train/train_pointnet.yaml start
seml pointcloud_sample add seml/configs/point_clouds/sample_cert/sample_cert_dgcnn.yaml start
seml pointcloud_sample add seml/configs/point_clouds/sample_cert/sample_cert_pointnet.yaml start
seml pointcloud_cert add seml/configs/point_clouds/cert/cert.yaml start
```
Then run `plotting/pointclouds/classification_certs.ipynb`.

### Attacks (Fig. 30a)
Run `plotting/attacks/pointnet/attack.ipynb`.  
Then run `plotting/attacks/pointnet/plot.ipynb`.

## Molecules

### SchNet (Fig. 9a)
```
seml molecule_train add seml/configs/force_fields/train_schnet_md17.yaml start
seml molecule_cert add seml/configs/force_fields/cert_schnet_md17.yaml start
```
Then run `plotting/force_fields/cert_plots_schnet.ipynb`.


### DimeNet++ (Fig. 4 & 8)
```
seml molecule_train add seml/configs/force_fields/train_dimenet_md17.yaml start
seml molecule_cert add seml/configs/force_fields/cert_dimenet_md17.yaml start
```
Then run `plotting/force_fields/cert_plots_schnet.ipynb`.

### SphereNet++ (Fig. 9b)
```
seml molecule_train add seml/configs/force_fields/train_spherenet_md17.yaml start
seml molecule_cert add seml/configs/force_fields/cert_spherenet_md17.yaml start
```
Then run `plotting/force_fields/cert_plots/schnet.ipynb`.

## Graphs

### Interval Bound Propagation (Fig. 12 & 13)
```
seml graph_cert_ibp add seml/configs/ibp/cert.yaml start
```
Then run `plotting/graphs/attributes_ibp/ged_certs.ipynb`.

### Convex Outer Adversarial Polytope (Fig. 6, 10, 11)
```
seml graph_cert_attributes_zuegner add seml/configs/attributes_zuegner/cert.yaml start
```
Then run `plotting/graphs/attributes_zuegner/ged_certs.ipynb`.

### Linearization & Dualization (Fig. 16 & 17)
```
seml graph_cert_structure_jin add seml/configs/structure_jin/cert.yaml start
```
Then run `plotting/graphs/structure_jin/ged_certs.ipynb`.


### Sparsity-Aware Smoothing (Fig. 5, 18-29)
TODO (configs are in `seml/configs/sparse_smoothing`)

### Policy iteration (Fig. 14 & 15)
TODO (configs are in `seml/configs/piPPNP`)

### Attacks (Fig. 30b)
Run `plotting/attacks/attribute_zuegner/attack.ipynb`.  
Then run `plotting/attacks/attributes_zuegner/plot.ipynb`.
