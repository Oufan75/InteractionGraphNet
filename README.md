# InteractionGraphNet
InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening


# Environment
```
conda create -f env.yml
```

# IGN Training (A toy example)
```
python ./codes/ign_train.py --gpuid 0 --epochs 5 --repetitions 3 --lr 0.0001 --l2 0.000001 --dropout 0.1 
```

# Binding Affinity Prediction 
We use the well-trained IGN model to predict the binding affinity of complexes generated from docking program

```
python3 ./codes/prediction.py --cpu True --num_process 12 --input_path  ./examples/ign_input
```
