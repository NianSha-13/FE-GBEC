**FE-GBEC**

------

we propose a deep ensemble clustering method. Initially, we employ the granular-ball approach for multi-granularity data representation, effectively reducing data complexity. Subsequently, we construct a self-enhanced association matrix based on the multi-granularity representation and iteratively refine feature representations by integrating this matrix with Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT). Improved the effectiveness of traditional ensemble clustering.



**Files**

------

These program mainly containing:

- A folder containing the experimental dataset is named "dataset".
- A folder called "gbutils" containing the code for the spheroidal fission.
- A folder called "LWEA_codes" containing the code for constructing the incidence matrix
- A folder called "measure" contains the code for calculating ARI, NMI and F-SCORE as well as the loss function
- seven python files

# Requirements

Installation requirements (Python 3.8)

- Pycharm
- Windows operating system
- scipy == 1.11.1
- numpy == 1.25.1
- scikit-learn == 1.3.0
- pandas == 2.0.3
- torch == 1.9.0+cu111
- torch-geometric == 2.0.0
- torchvision == 0.10.0+cu111

## Usage

Run Test_FE-GBEC.py to obtain the results of deep ensemble clustering based on multi-granularity representation.

