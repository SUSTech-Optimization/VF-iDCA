import numpy as np

def iteration_err(w, r, wp, rp):
    return np.sqrt(
        np.sum(np.square(w - wp)) + np.sum(np.square(r - rp))
    ) / np.sqrt(
        1 + np.sum(np.square(w)) + np.sum(np.square(r))
    )

def VF_iDCA(dclower, dcapp, DC_Setting = dict()):
    MAX_ITERATION = DC_Setting["MAX_ITERATION"] if "MAX_ITERATION" in DC_Setting.keys() else 10
    TOL = DC_Setting["TOL"] if "TOL" in DC_Setting.keys() else 1e-1
    dcapp.alpha.value = DC_Setting["alpha"] if "alpha" in DC_Setting.keys() else 1.
    dcapp.c_alpha = DC_Setting["c_alpha"] if "c_alpha" in DC_Setting.keys() else 1.
    dcapp.delta = DC_Setting["delta"] if "delta" in DC_Setting.keys() else 5. 
    r = DC_Setting["r"] if "r" in DC_Setting.keys() else np.ones(dcapp.rU.shape)
    w = DC_Setting["w"] if "w" in DC_Setting.keys() else np.zeros(dcapp.wU.shape)

    for _ in range(MAX_ITERATION):
        fL = dclower.solve(r)
        gamma = dclower.dual_value()
        dcapp.clare_variable(w, r)
        dcapp.clare_V(fL, gamma)
        _, wp, rp = dcapp.solve()

        err = iteration_err(w, r, wp, rp) 
        penalty = dcapp.violation.value

        if err < TOL and penalty < TOL:
            break

        if err * dcapp.alpha.value <= dcapp.c_alpha * min( 1., dcapp.alpha.value * dcapp.violation.value ):
            dcapp.alpha.value = dcapp.alpha.value + dcapp.delta

        w, r = wp, rp

    return wp, rp  

