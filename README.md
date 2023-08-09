# InteractionGraphNet
InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening

# Environment
```
conda env create -f env.yml
```

# IGN Training (retrained with LP-PDBBind)
LP-PDBBind https://github.com/THGLab/LP-PDBBind.git
```
python ./codes/ign_train.py --gpuid 0 --epochs 500 --batch_size 128 --graph_feat_size 128 --num_layers 2 --outdim_g3 128 --d_FC_layer 128 --repetitions 3 --lr 0.001 --l2 0.00001 --dropout 0.2
```

# Ligand Screening (on same target pocket)
We use the retrained IGN model to predict the binding affinity of complexes generated from docking program
```
python ./codes/score.py --protein xxx --ligands xxx --graph_dic_path input_path --output output.csv
```
