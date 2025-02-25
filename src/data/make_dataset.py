import sys
import os
from tabnanny import verbose
import numpy as np
import phate
import torch
from torch.utils.data import Dataset
import scipy
import scanpy as sc
import pickle
import scipy.io as sio
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import pytorch_lightning as pl
import scprep

from src import DATA_DIR


def rotation_transform(
    X : np.ndarray, # The input matrix, of size n x d (d is # dimensions)
    tilt_angles # a list of d-1 values in [0,2pi] specifying how much to tilt in d-1 the xy, yz (...) planes
):
    # Tilt matrix into arbitrary dimensions
    d = X.shape[1]
    assert len(tilt_angles) == d - 1
    # construct Tilting Matrices TM!
    tilting_matrices_tm = []
    for i in range(d-1):
        A = np.eye(d)
        A[i][i] = np.cos(tilt_angles[i])
        A[i+1][i+1] = np.cos(tilt_angles[i])
        A[i][i+1] = np.sin(tilt_angles[i])
        A[i+1][i] = - np.sin(tilt_angles[i])
        tilting_matrices_tm.append(A)
    X_tilted = X
    for tilter in tilting_matrices_tm:
        # print(X_tilted)
        X_tilted = X_tilted @ tilter
    return X_tilted, tilting_matrices_tm


def make_live_seq(PATH, emb_dim=20, knn=5, label=False):
    #adata_liveseq = sc.read_h5ad(os.path.join(PATH,"Liveseq.h5ad"))
    #adata_rnaseq = sc.read_h5ad(os.path.join(PATH,"scRNA.h5ad"))
    adata_liveseq = sc.read_h5ad(os.path.join(PATH,"adata_cancer_v2.h5ad"))
    X = adata_liveseq.X
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_live_seq = phate_operator.fit_transform(X)
    phate_live_seq = scipy.stats.zscore(phate_live_seq) 
    if label:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq, adata_liveseq.obs['celltype_treatment']
    else:
        return torch.tensor(X, requires_grad=True).float(), phate_live_seq


def make_n_sphere(n_obs=150, dim=3, emb_dim=2, knn=5):
    """Make an N-sphere with Muller's method. return a Tensor `requires_grad=True`."""
    norm = np.random.normal
    normal_deviates = norm(size=(dim, n_obs))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    X = (normal_deviates / radius).T
    # if train_dataset:
    #     phate_sphere = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_sphere = phate_operator.fit_transform(X)
    phate_sphere = scipy.stats.zscore(phate_sphere) 

    return torch.tensor(X, requires_grad=True).float(), phate_sphere



def make_n_sphere_two(n_obs=150, dim=10,emb_dim=2,knn=5):
    
    sphere = [] #Create sphere in 3D
    for i in range(n_obs):
        x = np.random.normal(0,1,3)
        sphere.append(x/(np.sqrt(np.sum(x**2))))

    nsphere = np.array(sphere)
    zerovec = np.zeros((n_obs,dim-3)) #add vector of zeros onto first 3 dimensions
    highdsphere = np.concatenate((nsphere,zerovec),axis=1) #Create high-d sphere
    
    deg = np.random.randint(0,360,1)[0]
    angles = list(np.repeat(deg,highdsphere.shape[1]-1)) #can insert angle you wish to rotate sphere by
    rotatesphere, _ = rotation_transform(highdsphere, angles)
    
    #run phate on rotated sphere
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    phate_sphere_rot = phate_operator.fit_transform(rotatesphere)
    phate_sphere_rot = scipy.stats.zscore(phate_sphere_rot)
    
    return torch.tensor(rotatesphere, requires_grad=True).float(), phate_sphere_rot


def make_tree(n_obs=150, dim=10, emb_dim=2, knn=5):
    """Make a tree dataset. Return a Tensor `requires_grad=True` and tree_phate"""

    n_branch = 8
    branch_length = int(n_obs/n_branch)
    tree_data, tree_clusters = phate.tree.gen_dla(n_dim=10, n_branch=n_branch, branch_length=branch_length)
    # if train_dataset:
    #     tree_phate = None
    # else:
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    tree_phate = phate_operator.fit_transform(tree_data)
    tree_phate = scipy.stats.zscore(tree_phate) 

    return torch.tensor(tree_data, requires_grad=True).float(), tree_phate, tree_clusters



def make_ipsc(n_obs=150,emb_dim=2,knn=5,indx=None):
    
    #load data
    print("loading data")
    initdir = os.getcwd()
    os.chdir(DATA_DIR)
    X = sio.loadmat('ipscData.mat')['data']
    os.chdir(initdir)
    ipsc_data = X[indx,:].squeeze()
    
    print("Performing Phate")
    phate_operator = phate.PHATE(random_state=42, verbose=True, n_components=emb_dim, knn=knn,t=250,decay=10)
    ipsc_phate = phate_operator.fit_transform(ipsc_data) #only compute on 100 points since it's so expensive for > 6000
    ipsc_phate = scipy.stats.zscore(ipsc_phate) 
    
    print("Done")
    return torch.tensor(ipsc_data, requires_grad=True).float(), ipsc_phate


def make_eb(n_obs = 150, emb_dim = 2, knn = 15, n_dim = 10, indx = None, download_path = "/home/Research/fm_geodesics/data/EB"):

    #download_path = "/home/ed667/project/sheaf-neural-network/data"
    if os.path.exists(os.path.join(download_path,"EB_phate.npy")):
        print("loading precomputed data")
        eb_phate = np.load(os.path.join(download_path,"EB_phate.npy"))
        eb_data = np.load(os.path.join(download_path,"EB_data.npy"))
        eb_y = np.load(os.path.join(download_path,"EB_labels.npy"), allow_pickle = True)

        map_dict = {'Day 00-03': 0,
             'Day 06-09': 1,
               'Day 12-15': 2,
                 'Day 18-21' : 3,
                   'Day 24-27': 4}

        eb_y = np.array([map_dict[i] for i in eb_y])[:,None]

        rng = np.random.default_rng(seed=421)
        if n_obs == -1:
            sel_idx = np.arange(len(eb_phate))
        else:
            sel_idx = rng.choice(np.arange(len(eb_phate)),n_obs,replace = False)
 
        #return torch.tensor(eb_data[sel_idx], requires_grad=True).float(), eb_phate[sel_idx], eb_y[sel_idx]
        return torch.tensor(eb_data[sel_idx], requires_grad=True).float(), eb_y[sel_idx]
        #return eb_phate[sel_idx], eb_y[sel_idx]
    
    else:

        print("loading raw data ....")
        sparse=True
        T1 = scprep.io.load_10X(os.path.join(download_path, "EB_raw", "T0_1A"), sparse=sparse, gene_labels='both')
        T2 = scprep.io.load_10X(os.path.join(download_path, "EB_raw",  "T2_3B"), sparse=sparse, gene_labels='both')
        T3 = scprep.io.load_10X(os.path.join(download_path, "EB_raw",  "T4_5C"), sparse=sparse, gene_labels='both')
        T4 = scprep.io.load_10X(os.path.join(download_path, "EB_raw",  "T6_7D"), sparse=sparse, gene_labels='both')
        T5 = scprep.io.load_10X(os.path.join(download_path, "EB_raw",  "T8_9E"), sparse=sparse, gene_labels='both')
        T1.head()

        filtered_batches = []
        for batch in [T1, T2, T3, T4, T5]:
            batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
            batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
            filtered_batches.append(batch)
        del T1, T2, T3, T4, T5 # removes objects from memory

        EBT_counts, sample_labels = scprep.utils.combine_batches(
            filtered_batches, 
            ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
            append_to_cell_names=True
        )
        del filtered_batches # removes objects from memory


        EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)
        EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)
        mito_genes = scprep.select.get_gene_set(EBT_counts, starts_with="MT-") # Get all mitochondrial genes. There are 14, FYI.
        EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
            EBT_counts, sample_labels, genes=mito_genes, 
            percentile=90, keep_cells='below')
        EBT_counts = scprep.transform.sqrt(EBT_counts)
        
        print("Performing PHATE")
        phate_operator = phate.PHATE(k = knn)
        eb_phate = phate_operator.fit_transform(EBT_counts)

        #perform PCA
        print("Performing PCA")
        pca = PCA(n_components=n_dim)
        #pca.fit(EBT_counts.values)
        #eb_data = EBT_counts.values @ pca.components_.T
        eb_data = pca.fit_transform(EBT_counts.to_numpy())

        
        eb_phate = scipy.stats.zscore(eb_phate)
        eb_data = eb_data/(np.max(eb_data)-np.min(eb_data))
        eb_phate = eb_phate/(np.max(eb_phate)-np.min(eb_phate))
        

        np.save(os.path.join(download_path, "EB_phate.npy"), eb_phate)
        np.save(os.path.join(download_path, "EB_data.npy"), eb_data)
        np.save(os.path.join(download_path, "EB_labels.npy"), sample_labels)
        
        print("Done")
        
        #return make_eb(n_obs = n_obs, emb_dim = emb_dim, knn =knn, n_dim = n_dim, indx = indx, download_path = download_path)
        return torch.tensor(eb_data, requires_grad=True).float(), eb_phate


def make_pbmc(n_obs=150,emb_dim=2,knn=5,indx=None):
    
    #extract PMBC data size/dimensions
    
    #load data
    print("loading data")
    initdir = os.getcwd()
    os.chdir(DATA_DIR) 
    with open('pbmc.pickle','rb') as f:
        X = pickle.load(f).values.squeeze()
    os.chdir(initdir)
    if indx is not None:
        iX = X[indx,:]
    else:
        iX = X 
    #perform PCA
    print("Performing PCA")
    pca = PCA(n_components=10)
    pca.fit(iX)
    pbmc_data = iX @ pca.components_.T

    print("Performing PHATE")
    phate_operator = phate.PHATE(random_state=42, verbose=True, n_components=emb_dim, knn=knn)    
    pbmc_phate = phate_operator.fit_transform(pbmc_data)
    pbmc_phate = scipy.stats.zscore(pbmc_phate) 
    
    print("Done")
    return torch.tensor(pbmc_data, requires_grad=True).float(), pbmc_phate

def make_swiss(n_obs=150, emb_dim=2,knn=5):
    
    swiss_data = sklearn.datasets.make_swiss_roll(n_samples=n_obs)[0]
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=emb_dim, knn=knn)
    swiss_phate = phate_operator.fit_transform(swiss_data) #only compute on 100 points since it's so expensive for > 6000
    swiss_phate = scipy.stats.zscore(swiss_phate) 
    
    return torch.tensor(swiss_data, requires_grad=True).float(), swiss_phate

def make_swiss_exp():
    
    swiss_data = np.load("swiss_roll_exp.npy")
    phate_operator = phate.PHATE(random_state=42, verbose=False, n_components=2, knn=3)
    swiss_phate = phate_operator.fit_transform(swiss_data) #only compute on 100 points since it's so expensive for > 6000
    swiss_phate = scipy.stats.zscore(swiss_phate) 
    return torch.tensor(swiss_data,requires_grad=True).float(), swiss_phate
    

class torch_dataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        target = self.Y[index, :]
        sample = self.X[index, :]
        return sample, target


class PLDataModule(pl.LightningDataModule):
    def __init__(self, name, n_obs, dim, emb_dim, batch_size, knn, PATH=None,indx=None):
        super().__init__()
        self.name = name
        self.n_obs = n_obs
        self.dim = dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.knn = knn
        self.PATH = PATH
        self.indx = indx

    def prepare_data(self) -> None:
        self.train_data = get_train_data(self.name, self.n_obs, self.dim, self.emb_dim, self.knn, self.PATH, self.indx)


    def setup(self, stage: str):
        return
        #self.train_loader = train_dataloader(self.name, self.n_obs, self.dim, self.emb_dim, self.batch_size, self.knn, self.PATH, self.indx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.train_data, batch_size = self.batch_size, shuffle = True,
                                           num_workers = 0, pin_memory = True)




def get_train_data(name, n_obs, dim, emb_dim, batch_size, knn, PATH=None,indx=None, data_seed = 0, download_path = None, **kwargs):
    """Create a Torch data loader for training."""


    # TODO: add warning if `name` is not implemented.
    if name.lower() == "sphere":
        #X, Y = make_n_sphere_two(n_obs, dim, emb_dim, knn)
        #Y = torch.tensor(Y).float()
        #train_dataset = torch_dataset(X, Y)
        #train_loader = torch.utils.data.DataLoader(
        #    dataset=train_dataset, batch_size=batch_size, shuffle=True
        #)
        rng = np.random.default_rng(seed=data_seed)

        X = torch.cat([torch.Tensor(rng.random((n_obs,1))) * 2* torch.pi - torch.pi,torch.Tensor(rng.random((n_obs,1))) * 0.5*(torch.pi) ],dim=1)
        Y = torch.zeros(n_obs,1)
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )


    elif name.lower() == "tree":
        X, Y, _ = make_tree(n_obs, dim, emb_dim, knn)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "live_seq":
        X, Y = make_live_seq(PATH, emb_dim, knn, label=False)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "ipsc":
        X, Y = make_ipsc(n_obs=n_obs,emb_dim=2,knn=5,indx=indx)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
    
    elif name.lower() == "eb":
        X, x_phate, Y = make_eb(n_obs=n_obs,emb_dim=2,knn=5,indx=indx, download_path = download_path)
        Y = torch.tensor(Y).float()
        x_phate = torch.tensor(x_phate).float()
        Y = torch.cat((Y,x_phate),-1)

        train_dataset = torch_dataset(X.detach(), Y.detach())
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "pbmc":
        X, Y = make_pbmc(n_obs=n_obs,emb_dim=2,knn=5,indx=indx)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory = True
        )
        
    elif name.lower() == "swiss_roll":
        X, Y = make_swiss(n_obs=n_obs,emb_dim=2,knn=5)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "swiss_roll_exp":
        X, Y = make_swiss_exp()
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
        
        
    return train_dataset

class torch_dataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        target = self.Y[index, :]
        sample = self.X[index, :]
        return sample, target


def train_dataloader(name, n_obs, dim, emb_dim, batch_size, knn, PATH=None,indx=None, download_path = None):
    """Create a Torch data loader for training."""

    # TODO: add warning if `name` is not implemented.
    if name.lower() == "sphere":
        X, Y = make_n_sphere_two(n_obs, dim, emb_dim, knn)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "tree":
        X, Y, _ = make_tree(n_obs, dim, emb_dim, knn)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    elif name.lower() == "live_seq":
        X, Y = make_live_seq(PATH, emb_dim, knn, label=False)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "ipsc":
        X, Y = make_ipsc(n_obs=n_obs,emb_dim=2,knn=5,indx=indx)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

            
    elif name.lower() == "eb":
        X, Y = make_eb(n_obs=n_obs, emb_dim=2,knn=5, n_dim = dim, indx=indx, download_path= download_path)
        Y = torch.tensor(Y).float()
        X = torch.tensor(X).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "pbmc":
        X, Y = make_pbmc(n_obs=n_obs,emb_dim=2,knn=5,indx=indx)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "swiss_roll":
        X, Y = make_swiss(n_obs=n_obs,emb_dim=2,knn=5)
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
    elif name.lower() == "swiss_roll_exp":
        X, Y = make_swiss_exp()
        Y = torch.tensor(Y).float()
        train_dataset = torch_dataset(X, Y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        
        
        
    return train_loader
# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
