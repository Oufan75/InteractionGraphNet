# File Name: graph_constructor.py
# E-mail: jiang_dj@zju.edu.cn
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
from scipy.spatial import distance_matrix
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot, BaseBondFeaturizer
    
import pickle
import os
import numpy as np
from functools import partial
import warnings
from pathlib import Path
import multiprocessing
from itertools import repeat
warnings.filterwarnings('ignore')


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()



def graphs_ligand_screen(ligand, protein, graph_dic_path, dis_threshold=8.0, path_marker='\\'):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """
    add_self_loop = False
    if not os.path.exists(ligand):
        print(f"{ligand} doesn't exist")
        return
    mol1 = Chem.SDMolSupplier(ligand)[0]
    mol2 = Chem.MolFromPDBFile(protein)
    key = Path(ligand).stem
    label = 0
    if mol1 is None or mol2 is None:
        print(f"{ligand} graph generation error")
        return
    
    # construct graphs1
    g = dgl.DGLGraph()
    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u + num_atoms_m1)
        dst2.append(v + num_atoms_m1)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g.add_edges(src_ls2, dst_ls2)

    # add interaction edges, only consider the euclidean distance within dis_threshold
    g3 = dgl.DGLGraph()
    g3.add_nodes(num_atoms)
    dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    node_idx = np.where(dis_matrix < dis_threshold)
    src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
    dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
    g3.add_edges(src_ls3, dst_ls3)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
    g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
    g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
    g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # efeats1
    g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g1_d = torch.cat([m1_d, m2_d])
    g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
    g3.edata['e'] = g3_d * 0.1

    # if add_3D:
    # init 'pos'
    g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
    g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
    g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    g.ndata.pop('pos')
    # detect the nan values in the D3_info_th
    if torch.any(torch.isnan(D3_info_th)):
        status = False
    else:
        status = True
    if status:
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)
            

def graphs_from_mol_mul(pdb_path, key, label, graph_dic_path, dis_threshold=8.0, path_marker='\\'):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """

    add_self_loop = False
    
    # use rdkit to read files
    pdbid = pdb_path.split("/")[-1]
    mol1 = Chem.SDMolSupplier(pdb_path + path_marker + f"{pdbid}_ligand.sdf")[0]
    #if mol1 is None:
    #    mol1 = Chem.MolFromMol2File(pdb_path + path_marker + f"{pdbid}_ligand.mol2")
    mol2 = Chem.MolFromPDBFile(pdb_path + path_marker + f"{pdbid}_protein.pdb")
    if mol1 is None:
        print(pdb_path + path_marker + "ligand.sdf")
        return
    if mol2 is None:
        print(pdb_path + path_marker + "protein.pdb")
        return
    
    # construct graphs1
    g = dgl.DGLGraph()
    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u + num_atoms_m1)
        dst2.append(v + num_atoms_m1)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g.add_edges(src_ls2, dst_ls2)

    # add interaction edges, only consider the euclidean distance within dis_threshold
    g3 = dgl.DGLGraph()
    g3.add_nodes(num_atoms)
    dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    node_idx = np.where(dis_matrix < dis_threshold)
    src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
    dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
    g3.add_edges(src_ls3, dst_ls3)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
    g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
    g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
    g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # efeats1
    g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g1_d = torch.cat([m1_d, m2_d])
    g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
    g3.edata['e'] = g3_d * 0.1

    # if add_3D:
    # init 'pos'
    g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
    g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
    g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    g.ndata.pop('pos')
    # detect the nan values in the D3_info_th
    if torch.any(torch.isnan(D3_info_th)):
        status = False
        print(key)
    else:
        status = True
    #except:
    #    g = None
    #    g3 = None
    #    status = False
    if status:
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'g': g, 'g3': g3, 'key': key, 'label': label}, f)



def collate_fn_v2(data_batch):
    graphs, graphs3, Ys, = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys


def collate_fn_v2_MulPro(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    graphs, graphs3, Ys, keys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, Ys, keys


def collate_fn_v2_2d(data_batch):
    graphs, graphs3, Ys, = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    bg.edata['e'] = bg.edata['e'][:, :11]  # only consider the 2d info for the ligand and protein graphs
    # mask the distance info in graphs3, this is compromised method
    bg3.edata['e'] = torch.zeros(bg3.number_of_edges(), 1)
    return bg, bg3, Ys


class GraphDatasetV2MulPro(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, keys, labels, data_dirs, graph_ls_path, graph_dic_path, num_process=6, dis_threshold=8.0,
                 add_3D=True, path_marker='\\'):
        """
        :param keys: the keys for the complexs, list
        :param labels: the corresponding labels for the complexs, list
        :param data_dirs: the corresponding data_dirs for the complexs, list
        :param graph_ls_path: the cache path for the total graphs objects (graphs.bin, graphs3.bin), labels, keys
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param num_process: the numer of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the atom-pair interactions
        :param add_3D: add the 3D geometric features to the edges of graphs
        :param path_marker: '\\' for windows and '/' for linux
        """
        self.origin_keys = keys
        self.origin_labels = labels
        self.origin_data_dirs = data_dirs
        self.graph_ls_path = graph_ls_path
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.add_3D = add_3D
        self.dis_threshold = dis_threshold
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.graph_ls_path+self.path_marker+'g.bin'):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_path + self.path_marker + 'g.bin', 'rb') as f:
                self.graphs = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'rb') as f:
                self.graphs3 = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'rb') as f:
                self.keys = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'rb') as f:
                self.labels = pickle.load(f)
        else:
            graph_dic_paths = repeat(self.graph_dic_path, len(self.origin_data_dirs))
            dis_thresholds = repeat(self.dis_threshold, len(self.origin_data_dirs))
            path_markers = repeat(self.path_marker, len(self.origin_data_dirs))

            print('Generate complex graph...')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(graphs_from_mol_mul,
                         zip(self.origin_data_dirs, self.origin_keys, self.origin_labels, graph_dic_paths,
                             dis_thresholds, path_markers))
            pool.close()
            pool.join()

            # collect the generated graph for each complex
            self.graphs = []
            self.graphs3 = []
            self.labels = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.graphs.append(graph_dic['g'])
                    self.graphs3.append(graph_dic['g3'])
                    self.labels.append(graph_dic['label'])
            # store to the disk
            with open(self.graph_ls_path + self.path_marker + 'g.bin', 'wb') as f:
                pickle.dump(self.graphs, f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'wb') as f:
                pickle.dump(self.graphs3, f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'wb') as f:
                pickle.dump(self.keys, f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'wb') as f:
                pickle.dump(self.labels, f)
            # stat complexes failed to make
            failed = [str(key) for key in self.origin_keys if key not in self.keys]
            with open(self.graph_ls_path + self.path_marker + 'failed.txt', 'w+') as f:
                f.write("\n".join(failed))
        # delete the temporary files
        cmdline = 'rm -rf %s' % (self.graph_dic_path + self.path_marker + '*')  # graph_dic_path
        os.system(cmdline)
        
    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)


class GraphDataset_iMiner(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, ligands, protein, graph_dic_path, num_process=6, dis_threshold=8.0,
                 add_3D=True, path_marker='\\'):
        """
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param num_process: the numer of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the atom-pair interactions
        :param add_3D: add the 3D geometric features to the edges of graphs
        :param path_marker: '\\' for windows and '/' for linux
        """
        self.ligands = ligands
        self.protein = protein
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.add_3D = add_3D
        self.dis_threshold = dis_threshold
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        graph_dic_paths = repeat(self.graph_dic_path, len(self.ligands))
        dis_thresholds = repeat(self.dis_threshold, len(self.ligands))
        path_markers = repeat(self.path_marker, len(self.ligands))
        protein = repeat(self.protein, len(self.ligands))
        print('IGN: generate complex graph...')

        pool = multiprocessing.Pool(self.num_process)
        pool.starmap(graphs_ligand_screen,
                     zip(self.ligands, protein, graph_dic_paths, dis_thresholds, path_markers))
        pool.close()
        pool.join()

        # collect the generated graph for each complex
        self.graphs = []
        self.graphs3 = []
        self.labels = []
        self.keys = os.listdir(self.graph_dic_path)

        for key in self.keys:
            with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                graph_dic = pickle.load(f)
                self.graphs.append(graph_dic['g'])
                self.graphs3.append(graph_dic['g3'])
                self.labels.append(graph_dic['label'])
        # delete the temporary files
        cmdline = 'rm -rf %s' % (self.graph_dic_path + self.path_marker + '*')  # graph_dic_path
        os.system(cmdline)
        
    def __getitem__(self, indx):
        return self.graphs[indx], self.graphs3[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)


