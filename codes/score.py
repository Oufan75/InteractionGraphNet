from graph_constructor import *
from utils import *
from model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import warnings
import argparse
import torch
import pandas as pd
import os
from dgl.data.utils import split_dataset

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
dgl.use_libxsmm(False)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_eval_epoch(model, validation_dataloader, device):
    pred = []
    keys = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, Ys, Keys = batch
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            outputs = model(bg, bg3)
            pred.extend(outputs.data.cpu().numpy().flatten().tolist())
            keys.extend(list(Keys))
    return pred, keys


batch_size = 512
dis_threshold = 12
# paras for model
node_feat_size = 40
edge_feat_size_2d = 12
edge_feat_size_3d = 21
graph_feat_size = 128
num_layers = 2
outdim_g3 = 128
d_FC_layer, n_FC_layer = 128, 2 #200
path_marker = '/'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ligands', type=str)
    argparser.add_argument('--protein', type=str)
    argparser.add_argument('--graph_dic_path', type=str, default='./examples/graph_dic_path',
                           help="path for storing graph dictionary objects (temporary files)")
    argparser.add_argument('--model_path', type=str, default='/global/scratch/users/ozhang/InteractionGraphNet/model_save/2023-06-23_00_36_38_49161.pth',
                           help="path for storing pretrained model")
    argparser.add_argument('--device', default="cpu",
                           help="the gpu id for the prediction or cpu")
    argparser.add_argument('--num_process', type=int, default=8,
                           help="the number of process for generating graph objects")
    argparser.add_argument('--output', type=str, default="/tmp/ign_result.csv",
                           help="output path")
    args = argparser.parse_args()
    graph_dic_path, model_path, device, num_process, ligand_file, protein, output = args.graph_dic_path, \
                                                                            args.model_path, \
                                                                            args.device, \
                                                                            args.num_process, \
                                                                            args.ligands, \
                                                                            args.protein, \
                                                                            args.output
    with open(ligand_file, "r") as f:
        ligands = f.read().split("\n")
        
    # generating the graph objective using multi process
    test_dataset = GraphDataset_iMiner(ligands, protein, graph_dic_path=graph_dic_path,
                                num_process=num_process, dis_threshold=dis_threshold, path_marker=path_marker)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=0,
                                collate_fn=collate_fn_v2_MulPro)
                                  
    DTIModel = DTIPredictorV4_V2(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                                graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                                d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=0.2, n_tasks=1)
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device("cpu")
    DTIModel.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    test_pred, key = run_a_eval_epoch(DTIModel, test_dataloader, device)

    res = pd.DataFrame({'ligand_names': key, 'ign_score': test_pred}) 
    res.to_csv(output, index=False)
