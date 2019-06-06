import numpy as np

def paran(rows, cols, iterations=50):

    def eigenvalues(data):
        cor = np.corrcoef(data)
        evs, _ = np.linalg.eig(cor)
        return evs[evs.argsort()[::-1]]
    
    def rand_pca_eigenvalues(rows, cols):
        sim_data = np.random.randn(rows, cols)
        evs = eigenvalues(sim_data.transpose())
        return evs

    for i in range(iterations):
        yield rand_pca_eigenvalues(rows, cols)