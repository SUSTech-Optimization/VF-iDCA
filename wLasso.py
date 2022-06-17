#%%
from VF_iDCA import VF_iDCA
import numpy as np 
import cvxpy as cp

#%%
class DC_lower():
    def __init__(self):
        self.wL = cp.Variable(p)
        self.rL = cp.Parameter(p, nonneg=True)
        LSTr = cp.sum_squares(XTr @ self.wL - yTr) 
        self.cons = [self.wL[i] <= self.rL[i] for i in range(p)]
        self.dc_lower = cp.Problem(cp.Minimize(LSTr), self.cons)

    def solve(self, r):
        self.rL.value = r
        result = self.dc_lower.solve(solver = cp.ECOS)
        return result

    def dual_value(self):
        return np.array([float(self.cons[i].dual_value) for i in range(p)])

class DC_approximated():
    def __init__(self):
        self.rho = 1.
        self.delta = 5.
        self.c_alpha = 1. 

        self.wU = cp.Variable(p)
        self.rU = cp.Variable(p)
        self.fL = cp.Parameter()
        self.wk = cp.Parameter(p)
        self.rk = cp.Parameter(p, nonneg=True)
        self.gamma = cp.Parameter(p)
        self.alpha = cp.Parameter(nonneg=True)

        self.alpha.value = 1.

        LSVal = cp.sum_squares(XVal @ self.wU - yVal)
        prox = cp.sum_squares(self.wU - self.wk) + cp.sum_squares(self.rU - self.rk)
        Vk = cp.sum_squares(XTr @ self.wU - yTr) - self.fL + self.gamma @ (self.rU - self.rk)
        self.violation = cp.maximum(*([0, Vk] + [cp.abs(self.wU[i]) - self.rU[i] for i in range(p)]))

        phi_k = LSVal + self.rho/2 * prox + self.alpha * self.violation
        bi_cons = [self.rU >= 0]

        self.dc_app = cp.Problem(cp.Minimize(phi_k), bi_cons)
    
    def solve(self):
        self.dc_app.solve(solver = cp.ECOS)
        return self.dc_app.value, self.wU.value, np.maximum(0, self.rU.value)
    
    def clare_variable(self, w, r):
        self.wk.value = w 
        self.rk.value = r
    
    def clare_V(self, fL, gamma):
        self.fL.value = fL
        self.gamma.value = gamma

#%%
# Generate Data
np.random.seed(42) # for reproduction

n, p = 600, 100
beta_nonzero = np.array([1,2,3,4,5])
beta_real = np.concatenate([beta_nonzero, np.zeros(p - len(beta_nonzero))])
X = np.random.randn(n, p)
y_true = X @ beta_real

# add noise
snr = 2
epsilon = np.random.randn(n)
SNR_factor = snr / np.linalg.norm(y_true) * np.linalg.norm(epsilon)
y = y_true + 1.0 / SNR_factor * epsilon

# split data
nTr, nVal = 100, 100
XTr, XVal, XTest = X[0:nTr], X[nTr:nTr + nVal], X[nTr + nVal:]
yTr, yVal, yTest = y[0:nTr], y[nTr:nTr + nVal], y[nTr + nVal:]

## define err function
def ls_err(X, y, w):
    return np.sum(np.square(X @ w - y))/len(y)

def err_report(w):
    print("%20s %8.3f" % ("train err:", ls_err(XTr, yTr, w)))
    print("%20s %8.3f" % ("validate err:", ls_err(XVal, yVal, w)))
    print("%20s %8.3f" % ("test err:", ls_err(XTest, yTest, w)))

#%%
w = np.zeros(p)
r = np.ones(p)
err_report(w)

dclower = DC_lower()
dcapp = DC_approximated()
DC_Setting = {"w":w, "r":r}
w, r = VF_iDCA(dclower, dcapp)
err_report(w)