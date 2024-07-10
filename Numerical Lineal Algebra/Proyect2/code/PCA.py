import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def compute_PCA(X):
    n = X.shape[0]
    Y = (1.0 / np.sqrt(n-1)) * X.T

    _, S, Vt = np.linalg.svd(Y, full_matrices=False)
    var_T = np.sum(S**2)
    v_exp = (S**2)/var_T
    cum_sum = np.cumsum(v_exp)
    for i in range(S.shape[0]):
        print(f"\tPortion of variance explained by the first {i+1} components: {cum_sum[i]}")
        print(f"\t\tStandard Deviation for component {i+1}: {S[i]}")
    return Vt@X, v_exp


## Dataset 1: example.dat
x_example = np.genfromtxt('example.dat', delimiter= ' ')

# covariance
print("Calculating PCA for example.dat\n")
print("with covariance matrix...")
pca_data, v_exp = compute_PCA(x_example.T)
plt.xlabel(f"PC1: {v_exp[0]*100:0.10f} % variance")
plt.ylabel(f"PC2: {v_exp[1]*100:0.10f} % variance")
plt.tick_params(direction='in')
plt.ticklabel_format(style='sci', scilimits=[0,0])
plt.scatter(pca_data[0,:],pca_data[1,:], marker='+', c="black")
plt.show()
print()
# correlation
print("with correlation matrix...")
new_A = x_example - np.mean(x_example, axis=0)
pca_data, v_exp = compute_PCA(new_A.T)
plt.xlabel(f"PC1: {v_exp[0]*100:0.10f} % variance")
plt.ylabel(f"PC2: {v_exp[1]*100:0.10f} % variance")
plt.tick_params(direction='in')
plt.ticklabel_format(style='sci', scilimits=[0,0])
plt.scatter(pca_data[0,:],pca_data[1,:], marker='+', c="black")
plt.show()
print()
print("\tsaved to pca_data_1.csv")
print()
np.savetxt("pca_data_1.csv", pca_data.T, delimiter=",")

## Dataset 2: RCsGoff.csv

x_RCsGoff = np.genfromtxt('RCsGoff.csv', delimiter= ',',dtype=str)
samples = x_RCsGoff[0, 1:].astype(str)
x_RCsGoff = x_RCsGoff[1:, 1:].T.astype(float)

print("Calculating PCA for RCsGoff.csv\n")
new_A = x_RCsGoff - np.mean(x_RCsGoff, axis=0)

pca_data, v_exp = compute_PCA(new_A.T)
plt.xlabel(f"PC1: {v_exp[0]*100:0.10f} % variance")
plt.ylabel(f"PC2: {v_exp[1]*100:0.10f} % variance")
plt.tick_params(direction='in')
plt.ticklabel_format(style='sci', scilimits=[0,0])
plt.scatter(pca_data[0,:],pca_data[1,:], marker='+', c="black")
plt.show()
print()
print("\tsaved to pca_data_RCsGoff.csv")
print()
df = pd.DataFrame(np.concatenate((np.array([samples]).T, pca_data, np.array([v_exp]).T), axis=1))
df.to_csv("pca_data_RCsGoff.csv")