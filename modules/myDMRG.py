# Module for myDMRG

import numpy as np

from tenpy.linalg import np_conserved as npc
from tenpy.linalg.lanczos import lanczos
from tenpy.linalg.sparse import NpcLinearOperator
from tenpy.algorithms.truncation import svd_theta

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
    psi, model, chi_max, eps:
        See attributes

    Attributes
    ----------
    psi : SimpleMPS
        The current ground-state (approximation).
    model :
        The model of which the groundstate is to be calculated.
    chi_max, eps:
        Truncation parameters, see :func:`a_mps.split_truncate_theta`.
    LPs, RPs : list of np.Array[ndim=3]
        Left and right parts ("environments") of the effective Hamiltonian.
        ``LPs[i]`` is the contraction of all parts left of site `i` in the network ``<psi|H|psi>``,
        and similar ``RPs[i]`` for all parts right of site `i`.
        Each ``LPs[i]`` has legs ``vL wL* vL*``, ``RPS[i]`` has legs ``vR* wR* vR``
    """
    def __init__(self, psi, model, chi_max, eps):
        
        #assert psi.L == model.lat.mps_sites() #and psi.bc == model.bc  # ensure compatibility
        
        self.H_mpo = model.H_MPO
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        #self.Es = []
        
        # initialize left and right environment
        D = self.H_mpo.dim[0]   # IS IT CORRECT?
        chi = psi._B[0].shape[0]
        
        LP = np.zeros([chi, D, chi], dtype=float)  # vR wR vR*
        RP = np.zeros([chi, D, chi], dtype=float)  # vL* wL vL
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = npc.Array.from_ndarray(data_flat=LP, 
                                             legcharges=[self.psi._B[0].conj().get_leg('vL*'), # should have conj and then taken 'vL'
                                                         self.H_mpo.get_W(0).conj().get_leg('wL*'),
                                                         self.psi._B[0].get_leg('vL')],
                                             labels=['vR', 'wR', 'vR*'])
        self.RPs[-1] = npc.Array.from_ndarray(data_flat=RP, 
                                             legcharges=[self.psi._B[-1].get_leg('vR'), # should have conj and then taken 'vL'
                                                         self.H_mpo.get_W(-1).conj().get_leg('wR*'),
                                                         self.psi._B[-1].conj().get_leg('vR*')],
                                             labels=['vL*', 'wL', 'vL'])
        
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    def sweep(self):
        
        EL = ER = 0
        # sweep from left to right
        for i in range(self.psi.L - 2): # 
            #print(i)
            EL += self.update_bond(i)
            
        # sweep from right to left
        for i in range(self.psi.L - 2, 0, -1):
            ER += self.update_bond(i)
        
        print(EL, ER)

    def update_bond(self, i):
        
        j = (i + 1) % self.psi.L
        
        # Build H eff
        self.Heff = TwoSiteH(self, i)
        
        # Get theta
        th = self.psi.get_theta(i, n=2)
        
        # Contract with environment
        #th = self.Heff.matvec(th)
        #th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1]) # map to 2D
        
        # Diagonalize & find ground state
        E, th, N = lanczos(self.Heff,th)
        
        th = th.combine_legs([['vL', 'p0'], ['p1', 'vR']], qconj=[+1, -1]) # map to 2D
        old_A = self.psi.get_B(i, form='A')
        U, S, VH, err, renormalize = svd_theta(th,
                                               {'chi_max' : self.chi_max},
                                               qtotal_LR=[old_A.qtotal, None],
                                               inner_labels=['vR','vL'])
        
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

        
        
class TwoSiteH(NpcLinearOperator):
    r"""Class defining the two-site effective Hamiltonian for Lanczos.
    The effective two-site Hamiltonian looks like this::
            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.
    If `combine` is True, we define `LHeff` and `RHeff`, which are the contractions of `LP` with
    `W0`, and `RP` with `W1`, respectively.
    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    i0 : int
        Left-most site of the MPS it acts on.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site (when moving right) or right side (when moving left)
        into pipes. This reduces the overhead of calculating charge combinations in the
        contractions, but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    move_right : bool
        Whether the the sweep is moving right or left for the next update.
        Ignored for the :class:`TwoSiteH`.
    Attributes
    ----------
    i0 : int
        Left-most site of the MPS it acts on.
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and
        physical leg for the left site and right site into pipes. This reduces
        the overhead of calculating charge combinations in the contractions,
        but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    acts_on : list of str
        Labels of the state on which `self` acts. NB: class attribute.
        Overwritten by normal attribute, if `combine`.
    LHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian.
        Labels ``'(vR*.p0)', 'wR', '(vR.p0*)'`` for bra, MPO, ket.
    RHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian.
        Labels ``'(p1*.vL)', 'wL', '(p1.vL*)'`` for ket, MPO, bra.
    LP, W0, W1, RP : :class:`~tenpy.linalg.np_conserved.Array`
        Tensors making up the network of `self`.
    """
    length = 2
    acts_on = ['vL', 'p0', 'p1', 'vR']

    def __init__(self, eng, i0):
        #self.i0 = i0
        self.LP = eng.LPs[i0]
        self.RP = eng.RPs[i0 + 1]
        self.W0 = eng.H_mpo.get_W(i0).replace_labels(['p', 'p*'], ['p0', 'p0*'])
        # 'wL', 'wR', 'p0', 'p0*'
        self.W1 = eng.H_mpo.get_W(i0 + 1).replace_labels(['p', 'p*'], ['p1', 'p1*'])
        # 'wL', 'wR', 'p1', 'p1*'
        self.dtype = eng.H_mpo.dtype
        #self.N = (self.LP.get_leg('vR').ind_len * self.W0.get_leg('p0').ind_len *
        #          self.W1.get_leg('p1').ind_len * self.RP.get_leg('vL').ind_len)

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.
        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, p0, p1, vR`` if combine=False, ``(vL.p0), (p1.vR)`` if True
        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        labels = theta.get_leg_labels()
        theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
        theta = npc.tensordot(self.W0, theta, axes=[['wL', 'p0*'], ['wR', 'p0']])
        theta = npc.tensordot(theta, self.W1, axes=[['wR', 'p1'], ['wL', 'p1*']])
        theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
        theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta
