# Module for myDMRG

import numpy as np
from tqdm import tqdm

from tenpy.linalg import np_conserved as npc
from tenpy.linalg.lanczos import lanczos
from tenpy.linalg.sparse import NpcLinearOperator
from tenpy.algorithms.truncation import svd_theta
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import BosonSite


# DMRG ENGINE
class SimpleDMRGEngine_Boson:
    """DMRG algorithm, implemented as class holding the necessary data.


        .->-vR            p*         vL->-.
        |                 ^               |
        |                 |               |
        (LP)->-wR   wL->-(W1)->-wR  wL->-(RP)
        |                 |               |
        |                 ^               |
        .-<-vR*           p         vL*-<-.

    Parameters
    ----------
    psi, model, chi_max:
        See attributes

    Attributes
    ----------
    psi : SimpleMPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated.
    chi_max:
        Truncation parameter.
    LPs, RPs : list of npc.Array
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vR wR vR*``, ``RPs[i]`` has legs ``vL wL vL*``
    Es : list of floats
        Energies of the ground state computed in each sweep
    """
    def __init__(self, psi, model, chi_max):

        self.H_mpo = model.H_MPO
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.Es = []

        # initialize left and right environment
        DL = self.H_mpo.get_W(0).shape[0]
        DR = self.H_mpo.get_W(-1).shape[1]
        chi = psi._B[0].shape[0]

        LP = np.zeros([chi, DL, chi], dtype=float)  # vR wR vR*
        RP = np.zeros([chi, DR, chi], dtype=float)  # vL* wL vL
        LP[:, 0, :] = np.eye(chi)
        RP[:, DR - 1, :] = np.eye(chi)
        self.LPs[0] = npc.Array.from_ndarray(data_flat=LP,
                                             legcharges=[self.psi._B[0].conj().get_leg('vL*'),
                                                         self.H_mpo.get_W(0).conj().get_leg('wL*'),
                                                         self.psi._B[0].get_leg('vL')],
                                                         labels=['vR', 'wR', 'vR*'])
        self.RPs[-1] = npc.Array.from_ndarray(data_flat=RP,
                                             legcharges=[self.psi._B[-1].get_leg('vR'),
                                                         self.H_mpo.get_W(-1).conj().get_leg('wR*'),
                                                         self.psi._B[-1].conj().get_leg('vR*')],
                                                         labels=['vL*', 'wL', 'vL'])

        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    def sweep(self, desc_prog):

        EL = ER = 0
        # sweep from left to right
        for i in tqdm(range(self.psi.L - 2), leave=False, desc=str(desc_prog)+'# right sweep'):
            EL += self.update_bond(i)
        # sweep from right to left
        for i in tqdm(range(self.psi.L - 2, 0, -1), leave=False, desc=str(desc_prog)+'# left sweep'):
            ER += self.update_bond(i)
        return((EL+ER)/(2*(self.psi.L - 2)))

    def update_bond(self, i):

        j = (i + 1) % self.psi.L

        self.Heff = TwoSiteH(self, i)# Build Heff

        th = self.psi.get_theta(i, n=2)# Get theta

        # Diagonalize & find ground state
        lanczos_params = {
            'cutoff':0.001,
            'E_tol': 1.e-10,
            'N_cache':10,
            'N_min':10,
            'reortho':True
        }
        E, th, N = lanczos(self.Heff, th, lanczos_params)

        th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1]) # map to 2D
        old_A = self.psi.get_B(i, form='A')
        U, S, VH, err, renormalize = svd_theta(th,{'chi_max' : self.chi_max},
                                               qtotal_LR=[old_A.qtotal, None],inner_labels=['vR','vL'])

        # Manipulate and put back into MPS
        U.ireplace_label('(vL.p0)', '(vL.p)')
        VH.ireplace_label('(p1.vR)', '(p.vR)')
        A = U.split_legs(['(vL.p)'])
        B = VH.split_legs(['(p.vR)'])
        self.psi.set_B(i, A, form='A')
        self.psi.set_B(i+1, B, form='B')
        self.psi.set_SR(i, S)

        # Update Environment
        self.update_LP(i)
        self.update_RP(j)
        return E

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = (i - 1) % self.psi.L
        RP = self.RPs[i]  # vL* wL vL
        B = self.psi.get_B(i, form='B')  # vL p vR
        Bc = B.conj()  # vL* p* vR*
        W = self.H_mpo.get_W(i) # wL wR p p*
        RP = npc.tensordot(B, RP, axes=('vR', 'vL'))  # vL p [vR], [vL] wL vL*
        RP = npc.tensordot(RP, W, axes=(('p', 'wL'), ('p*', 'wR')))  # vL [p] [wL] vL*, wL [wR] p [p*]
        RP = npc.tensordot(RP, Bc, axes=(('vL*', 'p'), ('vR*', 'p*')))  # vL [vL*] wL [p], vL* [p*] [vR*]
        self.RPs[j] = RP  # vL wL vL*

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""

        j = (i + 1) % self.psi.L
        LP = self.LPs[i]  # vR wR vR*
        A = self.psi.get_B(i, form='A')  # vL p vR
        Ac = A.conj()  # vL* p* vR*
        W = self.H_mpo.get_W(i)  # wL wR i i*
        LP = npc.tensordot(LP, A, axes=('vR', 'vL'))  # [vR] wR vR*, [vL] i vR
        LP = npc.tensordot(W, LP, axes=(('wL', 'p*'), ('wR', 'p')))  # [wL] wR p [p*], vR [wR] [p] vR*
        LP = npc.tensordot(Ac, LP, axes=(('vL*', 'p*'), ('vR*', 'p')))  # [vL*] [p*] vR*, vR [p] [vR*] wR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)

    def run(self, params):
        self.max_sweep = params.get('max_sweep', 5)
        self.eps = params.get('eps', 1e-3)
        self.V = params.get('V', 1e-6)
        sweep_counter = 0
        last_last_E = 2e10
        last_E = 1e10
        e_counter = 0
        while True:
            sweep_counter += 1
            if (sweep_counter > self.max_sweep): break
            if (e_counter > 2): break
            last_last_E = last_E
            last_E = self.sweep(sweep_counter)
            self.Es.append(last_E)
            if (abs(last_E - last_last_E) <= self.eps*self.V):
                e_counter +=1


# EFFECTVE HAMITONIAN
class TwoSiteH(NpcLinearOperator):
    """The effective two-site Hamiltonian looks like this:
            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.
    Parameters
    ----------
    eng : :class:`~SimpleDMRGEngine_Boson`
        Engine that we defined
    i0 : int
        Left-most site of the MPS it acts on.
    Attributes
    ----------
    LP, W0, W1, RP : :class:`~tenpy.linalg.np_conserved.Array`
        Tensors making up the network of `self`.
    """
    length = 2
    acts_on = ['vL', 'p0', 'p1', 'vR']

    def __init__(self, eng, i0):
        self.LP = eng.LPs[i0]
        self.RP = eng.RPs[i0 + 1]
        self.W0 = eng.H_mpo.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        # 'wL', 'wR', 'p0', 'p0*'
        self.W1 = eng.H_mpo.get_W(i0 + 1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        # 'wL', 'wR', 'p1', 'p1*'
        self.dtype = eng.H_mpo.dtype

    def matvec(self, theta):

        labels = theta.get_leg_labels()
        theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
        theta = npc.tensordot(self.W0, theta, axes=[['wL', 'p0*'], ['wR', 'p0']])
        theta = npc.tensordot(theta, self.W1, axes=[['wR', 'p1'], ['wL', 'p1*']])
        theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)
        return theta


# MPO MODEL
class myModel(CouplingMPOModel):

    def init_sites(self, model_params):
        n_max = model_params.get('n_max', 0)
        filling = model_params.get('filling', 0)
        conserve = model_params.get('conserve', 'N')
        site = BosonSite(Nmax=n_max, conserve=conserve, filling=filling)
        return site

    def init_terms(self, model_params):
        rc = model_params.get('rc', 0)
        t = model_params.get('t', 1.)
        V = model_params.get('V', 0.)
        ryd = model_params.get('ryd', False)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t, u1, 'Bd', u2, 'B', dx, plus_hc=True)
        if ryd:
            for dx in range(1,2*(rc+1)):
                self.add_coupling(V/(1+(dx/rc)**6), 0, 'N', 0, 'N', dx)
        else:
            for dx in range(1,rc+1):
                self.add_coupling(V, 0, 'N', 0, 'N', dx)


# FUNCTIONS
def structure_factor(k, psi, model_params):

    L = model_params.get('L', 0)
    corr_matrix = psi.correlation_function('dN','dN')
    res = 0
    j = 0+1j
    for ii in range(L):
        for jj in range(L):
            res += (corr_matrix[ii,jj]/L) * np.exp(j*k*(ii-jj))
    return res

def g_2(psi, model_params):
    L = model_params.get('L', 0)
    corr_matrix = psi.correlation_function('N','N')
    out = np.zeros(L)
    norm = np.zeros(L)
    for ii in range(L):
        for jj in range(L):
            out[abs(ii-jj)] += corr_matrix[ii,jj]
            norm[abs(ii-jj)] += 1
    return out/norm

def B_1(psi, model_params):
    L = model_params.get('L', 0)
    corr_matrix = psi.correlation_function('Bd','B')
    out = np.zeros(L-1)
    norm = np.zeros(L-1)
    for jj in range(L):
        out[abs(1-jj)] += corr_matrix[1,jj]
        norm[abs(1-jj)] += 1
    return out/norm

def E_k(k, psi, model_params):
    L = model_params.get('L', 0)
    t = model_params.get('t', 0)
    corr_matrix = psi.correlation_function('Bd','B') + psi.correlation_function('B','Bd')
    res = 0
    for ii in range(L):
        res += corr_matrix[ii,(ii+1)%L]
    return res*(1 - np.cos(2*np.pi*k/L))*t/L

def n_k(k, psi, model_params):

    L = model_params.get('L', 0)
    corr_matrix = psi.correlation_function('Bd','B')
    res = 0
    j = 0+1j
    for ii in range(L):
        for jj in range(L):
            res += (corr_matrix[ii,jj]/L) * np.exp(j*k*(ii-jj))
    return res
