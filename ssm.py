import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.signal import convolve2d
import matplotlib.pyplot as plt, matplotlib as mpl
from Schelling_Segregation.utils import plot_red_lines
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
#Options
params = {'text.usetex' : True}
plt.rcParams.update(params)
CMAP = mpl.colors.ListedColormap([(1,1,1, 0),
                                  (1,0,0, 1),
                                  (0,0,1, 1)])
KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.int8)
N = 60        # Grid will be N x N
SIM_T = 0.4  # Similarity threshold (that is 1-τ)
EMPTY = 10   # Percentage of empty spots
B_to_R = 100  # Ratio of blue to red people

MAX_SIMULATIONS = 200
BOUNDARY = 'wrap' # wrap or fill
def rand_init(N, B_to_R, EMPTY):
    """
    WHITE = '0'
    BLACK = '1'
    EMPTY = '-1'
    """
    vacant = N * N * EMPTY // 100
    population = N * N - vacant
    blues = int(population * 1 / (1 + 100/B_to_R))
    reds = population - blues
    M = np.zeros(N*N, dtype=np.int8)
    M[:reds] = 1
    M[-vacant:] = -1
    np.random.shuffle(M)
    return M.reshape(N,N)

def evolve(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either fill, pad, or wrap
    If SIM_R < SIM_T, then the person moves to an empty house.
    """
    W_neighs = convolve2d(M == 0,  KERNEL, mode='same', boundary=boundary)
    B_neighs = convolve2d(M == 1,  KERNEL, mode='same', boundary=boundary)
    W_dissatified = (W_neighs / 8 < SIM_T) & (M==0)
    B_dissatified = (B_neighs / 8 < SIM_T) & (M==1)
    vacant = (M == -1).sum()
    N_W_dissatified = W_dissatified.sum()
    N_B_dissatified = B_dissatified.sum()
    if N_W_dissatified + N_B_dissatified > vacant:
        dissatisfied = ([0] * N_W_dissatified) + ([1] * N_B_dissatified)
        np.random.shuffle(dissatisfied)
        dissatisfied = dissatisfied[:vacant]
        N_B_dissatified = sum(dissatisfied)
        N_W_dissatified = vacant - N_B_dissatified

    B_moving_ = B_dissatified[B_dissatified]
    W_moving_ = W_dissatified[W_dissatified]
    B_moving_[N_B_dissatified:] = False
    W_moving_[N_W_dissatified:] = False
    np.random.shuffle(B_moving_)
    np.random.shuffle(W_moving_)
    B_dissatified[B_dissatified] = B_moving_
    W_dissatified[W_dissatified] = W_moving_
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:N_W_dissatified] = 0
    filling[N_W_dissatified:N_W_dissatified + N_B_dissatified] = 1
    np.random.shuffle(filling)
    M[(M==-1)] = filling
    M[W_dissatified + B_dissatified] = -1

def evolve2(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either fill, pad, or wrap
    If SIM_R < SIM_T, then the person moves to an empty house.
    """
    W_neighs = convolve2d(M == 0,  KERNEL, mode='same', boundary=boundary)
    B_neighs = convolve2d(M == 1,  KERNEL, mode='same', boundary=boundary)
    W_dissatified = (W_neighs / 8 < SIM_T) & (M==0)
    B_dissatified = (B_neighs / 8 < SIM_T) & (M==1)
    M[B_dissatified | W_dissatified] = - 1
    vacant = (M == -1).sum()
    N_W_dissatified, N_B_dissatified = W_dissatified.sum(), B_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:N_W_dissatified] = 0
    filling[N_W_dissatified:N_W_dissatified + N_B_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling

def evolve3(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or pad
    If SIM_R < SIM_T, then the person moves to an empty house.
    """
    kws = dict(mode='same', boundary=boundary)
    B_neighs = convolve2d(M == 0, KERNEL, **kws)
    R_neighs = convolve2d(M == 1, KERNEL, **kws)
    Neighs   = convolve2d(M != -1,  KERNEL, **kws)
    B_dissatified = (B_neighs / Neighs < SIM_T) & (M == 0)
    R_dissatified = (R_neighs / Neighs < SIM_T) & (M == 1)
    M[R_dissatified | B_dissatified] = - 1
    vacant = (M == -1).sum()
    N_B_dissatified, N_R_dissatified = B_dissatified.sum(), R_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:N_B_dissatified] = 0
    filling[N_B_dissatified:N_B_dissatified + N_R_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling
    return M

def evolve4(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or pad
    If SIM_R < SIM_T, then the person moves to an empty house.
    """
    kws = dict(mode='same', boundary=boundary)
    B_neighs = convolve2d(M == 0, KERNEL, **kws).astype(np.float16)
    R_neighs = convolve2d(M == 1, KERNEL, **kws).astype(np.float16)
    Neighs   = convolve2d(M != -1,  KERNEL, **kws).astype(np.float16)

    B_dissatified = (np.divide(B_neighs, Neighs, out=np.zeros_like(B_neighs), where=Neighs != 0) < SIM_T) & (M == 0)
    R_dissatified = (np.divide(R_neighs, Neighs, out=np.zeros_like(R_neighs), where=Neighs != 0) < SIM_T) & (M == 1)
    M[R_dissatified | B_dissatified] = - 1
    vacant = (M == -1).sum()
    N_B_dissatified, N_R_dissatified = B_dissatified.sum(), R_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:N_B_dissatified] = 0
    filling[N_B_dissatified:N_B_dissatified + N_R_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling
    return M


##### MORE WEBSITE PLOTS 1

# M = rand_init(N, B_to_R, EMPTY)
# plt.imshow(M, cmap=CMAP)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.yaxis.set_major_formatter(plt.NullFormatter())
# ax.set_xticks(np.arange(N) - .5)
# ax.set_yticks(np.arange(N) - .5)
# ax.grid()
# ax.tick_params(axis='both', which='both',
#                bottom=False, top=False,
#                left=False, right=False)
# plt.savefig('/Users/Luca/Downloads/schelling_grid_init.svg', transparent=True)
# plt.show()

##### MORE WEBSITE PLOTS 2
MAX_SIMULATIONS = 500
EQs = []
SIM_T_RANGE = [0.25, 0.4, 0.6, 0.75]
for SIM_T in SIM_T_RANGE:
    M = rand_init(N, B_to_R, EMPTY)
    for it in range(MAX_SIMULATIONS):
        if it > 2:
            old_M = np.copy(M)
        M = evolve3(M, boundary=BOUNDARY)
        if it>2 and (M == old_M).all():
            break
    EQs.append(M)


f, axs = plt.subplots(2, 2, figsize=(6, 6))
for ax, M, lb in zip(axs.flatten(), EQs, SIM_T_RANGE):
    ax.imshow(M, cmap=CMAP)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks(np.arange(N) - .5)
    ax.set_yticks(np.arange(N) - .5)
    ax.grid()
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False,
                   left=False, right=False)
    ax.set_title(r'$1-\tau={}$'.format(lb))
plt.tight_layout()
plt.savefig('/Users/Luca/Downloads/schelling_params.svg', transparent=True)
plt.show()

##### MORE WEBSITE PLOTS 3
MAX_SIMULATIONS = 500
SIM_T = 0.6
M = rand_init(N, B_to_R, EMPTY)
STATES = [np.copy(M)]
for it in range(MAX_SIMULATIONS):
    if it > 2:
        old_M = np.copy(M)
    M = evolve3(M, boundary=BOUNDARY)
    STATES.append(np.copy(M))
    if it>2 and (M == old_M).all():
        break

plt.imshow(M, cmap=CMAP)
plt.show()

for n, S in enumerate(STATES):
    plt.imshow(S, cmap=CMAP)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks(np.arange(N) - .5)
    ax.set_yticks(np.arange(N) - .5)
    ax.grid()
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False,
                   left=False, right=False)
    plt.tight_layout()
    plt.savefig(f'/Users/Luca/Downloads/{n}.svg', transparent=True)
    plt.show()

##### MORE WEBSITE PLOTS 4
EMPTY = 2
B_to_R = 100
MAX_SIMULATIONS = 500
SATISFACTION = []
SIM_T_RANGE = np.linspace(0, 1, 100)
for SIM_T in tqdm(SIM_T_RANGE):
    SATISFACTION.append([])
    for _ in range(5):  # Monte Carlo
        M = rand_init(N, B_to_R, EMPTY)
        for it in range(MAX_SIMULATIONS):
            if it > 2:
                old_M = np.copy(M)
            M = np.copy(evolve4(M, boundary=BOUNDARY))
            if it>2 and (M == old_M).all():
                break
        B_neighs = convolve2d(M == 0, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
        R_neighs = convolve2d(M == 1, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
        Neighs   = convolve2d(M != -1,  KERNEL, mode='same', boundary=BOUNDARY).astype(float)

        B_satisfaction = ((M == 0)*np.divide(B_neighs, Neighs, out=np.zeros_like(B_neighs), where=Neighs != 0)).sum()
        R_satisfaction = ((M == 1)*np.divide(R_neighs, Neighs, out=np.zeros_like(R_neighs), where=Neighs != 0)).sum()
        SATISFACTION[-1].append((B_satisfaction + R_satisfaction) / (N*N*(1-EMPTY//100)))
        # SATISFACTION[-1].append((B_satisfaction + R_satisfaction)/2)


S = np.array(SATISFACTION).mean(axis=1)
Se = np.array(SATISFACTION).std(axis=1)
plt.plot(SIM_T_RANGE, S)
plt.fill_between(SIM_T_RANGE, S - Se, S + Se, alpha=0.3)
plt.xlim(0,1)
plt.ylim(.49,1.01)
plt.show()
# 1000 points, 200MC
# pd.DataFrame(SATISFACTION).to_csv("./Schelling_Segregation/data/EMPTY10p_BRR_100.csv")
# pd.DataFrame(SATISFACTION).to_csv("./Schelling_Segregation/data/EMPTY10p_BRR_25.csv")
# pd.DataFrame(SATISFACTION).to_csv("./Schelling_Segregation/data/EMPTY2p_BRR_100.csv")

# TODO - temp: Fix SATISFACTION ... to be removed: has been fixed
# SATISFACTION = pd.read_csv("./Schelling_Segregation/data/EMPTY10p_BRR_100.csv", index_col=0).values
# SATISFACTION = (np.array(SATISFACTION) * 2) * N**2 / (N*N*(1-EMPTY/100))


# Repeat with different EMPTY and different B_to_R

## Plot all three together
import seaborn as sns
sns.set()
PATH = './Schelling_Segregation/data'
plt.figure(figsize=(6,5))
for file, (empty, brr), c in zip(['EMPTY10p_BRR_100', 'EMPTY10p_BRR_25', 'EMPTY2p_BRR_100'],
                           [[10, 100], [10, 25], [2, 100]],
                           ['tab:blue', 'tab:red', 'tab:green']):
    df = pd.read_csv(f'{PATH}/{file}.csv', index_col=0)
    S = df.mean(axis=1).values
    Se = df.std(axis=1).values
    plt.plot(SIM_T_RANGE, S,
             color=c, label=r'Vacant $= {}\%$, $R/B={}$'.format(empty, brr/100))
    plt.fill_between(SIM_T_RANGE, S-Se, S+Se, color=c, alpha=0.3)
    plt.xlim(0,1)
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-0.015, 1.3))
plt.xlabel(r'$1-\tau$')
plt.ylabel(r'$\langle S\rangle$', rotation=0, labelpad=15)
plt.gca().patch.set_alpha(0)
for s in ['top', 'bottom']:
    plt.gca().spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('/Users/Luca/Downloads/Schellign_Satisfaction.svg',
            transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()




##### MORE WEBSITE PLOTS 5: CONNECTED COMPONENTS
from Adjacency_for_SquareLattice import adjacency_from_square_lattice
import networkx as nx
EMPTY = 2
B_to_R = 100
MAX_SIMULATIONS = 200
N_CCs = []
SIM_T_RANGE = np.linspace(0, 1, 1000)
for SIM_T in tqdm(SIM_T_RANGE):
    N_CCs.append([])
    for _ in range(200):  # Monte Carlo
        M = rand_init(N, B_to_R, EMPTY)
        for it in range(MAX_SIMULATIONS):
            if it > 2:
                old_M = np.copy(M)
            M = np.copy(evolve4(M, boundary=BOUNDARY))
            if it>2 and (M == old_M).all():
                break
        df = pd.DataFrame(M)
        df[df==-1] = np.nan
        A = adjacency_from_square_lattice(df.fillna(method='pad').values,
                                          periodic_bc=True)
        CC = [*nx.connected_components(nx.from_numpy_array(A))]
        N_CCs[-1].append(len(CC))

# pd.DataFrame(N_CCs).to_csv("./Schelling_Segregation/data/ConnectedComponents_Default.csv")
# pd.DataFrame(N_CCs).to_csv("./Schelling_Segregation/data/ConnectedComponents_10p_BRR_25.csv")
# pd.DataFrame(N_CCs).to_csv("./Schelling_Segregation/data/ConnectedComponents_2p_BRR_100.csv")

## Plot all three together
import seaborn as sns
sns.set()
PATH = './Schelling_Segregation/data'
plt.figure(figsize=(6,5))
for file, (empty, brr), c in zip(['ConnectedComponents_Default',
                                  'ConnectedComponents_10p_BRR_25',
                                  'ConnectedComponents_2p_BRR_100'],
                           [[10, 100], [10, 25], [2, 100]],
                           ['tab:blue', 'tab:red', 'tab:green']):
    df = pd.read_csv(f'{PATH}/{file}.csv', index_col=0)
    S = df.mean(axis=1).values
    Se = df.std(axis=1).values
    plt.plot(SIM_T_RANGE, S,
             color=c, label=r'Vacant $= {}\%$, $R/B={}$'.format(empty, brr/100))
    plt.fill_between(SIM_T_RANGE, S-Se, S+Se, color=c, alpha=0.3)
    plt.xlim(0,1)
lgd = plt.legend(loc='upper left', bbox_to_anchor=(-0.015, 1.3))
plt.xlabel(r'$1-\tau$')
plt.ylabel(r'$\left|C\right|$', rotation=0, labelpad=15)
plt.gca().patch.set_alpha(0)
for s in ['top', 'bottom']:
    plt.gca().spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('/Users/Luca/Downloads/Schellign_CC.svg',
            transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()




##### MORE WEBSITE PLOTS 6: PHASE SPACE
N = 100
EMPTY = 10
B_to_R = 100
MAX_SIMULATIONS = 250
SATISFACTION = []
SIM_T_RANGE = np.linspace(0, 1, 201)
EMPTY_RANGE = np.linspace(0, 100, 101)
for SIM_T in tqdm(SIM_T_RANGE):
    SATISFACTION.append([])
    for EMPTY in EMPTY_RANGE:
        EMPTY = int(EMPTY)
        SATISFACTION[-1].append([])
        for _ in range(20):  # Monte Carlo
            M = rand_init(N, B_to_R, EMPTY)
            for it in range(MAX_SIMULATIONS):
                if it > 2:
                    old_M = np.copy(M)
                M = np.copy(evolve4(M, boundary=BOUNDARY))
                if it>2 and (M == old_M).all():
                    break
            B_neighs = convolve2d(M == 0, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
            R_neighs = convolve2d(M == 1, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
            Neighs   = convolve2d(M != -1,  KERNEL, mode='same', boundary=BOUNDARY).astype(float)

            B_satisfaction = ((M == 0)*np.divide(B_neighs, Neighs, out=np.zeros_like(B_neighs), where=Neighs != 0)).sum()
            R_satisfaction = ((M == 1)*np.divide(R_neighs, Neighs, out=np.zeros_like(R_neighs), where=Neighs != 0)).sum()
            SATISFACTION[-1][-1].append((B_satisfaction + R_satisfaction) / (N*N*(1-EMPTY//100)))


S = np.array(SATISFACTION).mean(axis=2)
# pd.DataFrame(S).to_csv("./Schelling_Segregation/data/HM100.csv")
# pd.DataFrame(S).to_csv("./Schelling_Segregation/data/HM200.csv")
# pd.DataFrame(S).to_csv("./Schelling_Segregation/data/HM200_2.csv")
S = pd.read_csv("./Schelling_Segregation/data/HM200_2.csv", index_col=0).values
S[:, 0] = S[:, 1]
S[:, 0] = S[:, 1] = S[:, 2]
#
from scipy import interpolate
EMPTY_RANGE = np.linspace(0, 100, 101)
f = interpolate.interp2d(SIM_T_RANGE, EMPTY_RANGE, S.T, kind='linear')
EMPTY_RANGE = np.linspace(0, 100, 401)
S = f(SIM_T_RANGE, EMPTY_RANGE).T



fig, axs = plt.subplots(2, 1, sharex=True,
                        gridspec_kw={'height_ratios': [1, 10]})
fig.subplots_adjust(hspace=0)  # Remove horizontal space between axes

plot_red_lines(ax=axs[0])
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 10)
axs[0].yaxis.set_major_formatter(plt.NullFormatter())
axs[0].tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False)
for s in ['right', 'left', 'top', 'bottom']:
    axs[0].spines[s].set_visible(False)

cmesh = axs[1].pcolormesh(SIM_T_RANGE, EMPTY_RANGE, S.T,
                          linewidth=0, rasterized=True,
                          vmin=0, vmax=1)
# plot_red_lines(ax=axs[1], alpha=0.1)
axs[1].set_xlim(0, 1)
cmesh.set_edgecolor('face')
axs[1].set_xlabel(r'$1-\tau$')
axs[1].set_ylabel(r'$N_v/N^2\ [\%]$')
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
cbar = plt.colorbar(cmesh, cax=cax, **kw)
cbar.ax.set_ylabel(r'$\langle S\rangle$', rotation=0, labelpad=10)
plt.savefig('/Users/Luca/Downloads/Schellign_HM.svg',
            transparent=True, bbox_inches='tight')
plt.show()


##### MORE WEBSITE PLOTS 7: PHASE SPACE 2 wrt B_to_R ratio
N = 100
EMPTY = 10
B_to_R = 100
MAX_SIMULATIONS = 250
SATISFACTION = []
SIM_T_RANGE = np.linspace(0, 1, 200)
B_to_R_RANGE = np.linspace(0, 100, 201)
for SIM_T in tqdm(SIM_T_RANGE):
    SATISFACTION.append([])
    for B_to_R in B_to_R_RANGE:
        SATISFACTION[-1].append([])
        for _ in range(10):  # Monte Carlo
            M = rand_init(N, B_to_R, EMPTY)
            for it in range(MAX_SIMULATIONS):
                if it > 2:
                    old_M = np.copy(M)
                M = np.copy(evolve4(M, boundary=BOUNDARY))
                if it>2 and (M == old_M).all():
                    break
            B_neighs = convolve2d(M == 0, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
            R_neighs = convolve2d(M == 1, KERNEL, mode='same', boundary=BOUNDARY).astype(float)
            Neighs   = convolve2d(M != -1,  KERNEL, mode='same', boundary=BOUNDARY).astype(float)

            B_satisfaction = ((M == 0)*np.divide(B_neighs, Neighs, out=np.zeros_like(B_neighs), where=Neighs != 0)).sum()
            R_satisfaction = ((M == 1)*np.divide(R_neighs, Neighs, out=np.zeros_like(R_neighs), where=Neighs != 0)).sum()
            SATISFACTION[-1][-1].append((B_satisfaction + R_satisfaction) / (N*N*(1-EMPTY//100)))


S = np.array(SATISFACTION).mean(axis=2)
pd.DataFrame(S).to_csv("./Schelling_Segregation/data/HM2.csv")
S = pd.read_csv("./Schelling_Segregation/data/HM2.csv", index_col=0).values
S[:, 0] = S[:, 1]




fig, axs = plt.subplots(2, 1, sharex=True,
                        gridspec_kw={'height_ratios': [1, 10]})
fig.subplots_adjust(hspace=0)  # Remove horizontal space between axes

plot_red_lines(ax=axs[0])
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 10)
axs[0].yaxis.set_major_formatter(plt.NullFormatter())
axs[0].tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False)
for s in ['right', 'left', 'top', 'bottom']:
    axs[0].spines[s].set_visible(False)

cmesh = axs[1].pcolormesh(SIM_T_RANGE, B_to_R_RANGE/100, S.T, linewidth=0, rasterized=True)
# plot_red_lines(ax=axs[1], alpha=0.1)
axs[1].set_xlim(0, 1)
cmesh.set_edgecolor('face')
axs[1].set_xlabel(r'$1-\tau$')
axs[1].set_ylabel(r'$\frac{B}{R}$', rotation=0, labelpad=10)
cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
cbar = plt.colorbar(cmesh, cax=cax, **kw)
cbar.ax.set_ylabel(r'$\langle S\rangle$', rotation=0, labelpad=10)
plt.savefig('/Users/Luca/Downloads/Schellign_HM2.svg',
            transparent=True, bbox_inches='tight')
plt.show()







if __name__=='__main__':
    SIM_T = 0.4
    mpl.use('TkAgg')
    M = rand_init(N, B_to_R, EMPTY)
    plt.close()

    for it in range(MAX_SIMULATIONS):
        if it > 2:
            old_M = np.copy(M)
            ax.clear()  # start removing points if you don't want all shown
        evolve4(M, boundary=BOUNDARY)
        if it > 2 and (M == old_M).all():
            break
        plt.imshow(M, cmap=CMAP)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        # ax.set_xticks(np.arange(N) - .5)
        # ax.set_yticks(np.arange(N) - .5)
        # ax.grid()
        # ax.tick_params(axis='both', which='both',
        #                bottom=False, top=False,
        #                left=False, right=False)

        # plt.title(f'Iteration {it}\n W={(M==0).sum()}, B={(M==1).sum()}, E={(M==-1).sum()}')
        plt.title(f'Iteration {it + 1} on {N}x{N} grid\n W/B={B_to_R}%, Empy={EMPTY}%, Similarity threshold={SIM_T}')
        plt.draw()
        plt.pause(0.0001)  # is necessary for the plot to update for some reason
plt.show()
