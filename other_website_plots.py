import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
#Options
params = {'text.usetex' : True}
plt.rcParams.update(params)
text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
               'fontweight': 'bold'}
Ngh_CLR = (85/255,203/255,253/255,0.5)
CMAP = mpl.colors.ListedColormap([(1,1,1,0), Ngh_CLR])

# Points:  M[0,0] = M[2,-2] = M[-2,1] = 1
M = np.zeros((8,8))
M[-4,1:4] = M[-2,1:4] = M[-3,1] = M[-3,3] = 1
M[0,-2:] = M[2,-2:] = M[1,-2] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7/2))
for n, (ax, title) in enumerate(zip((ax1, ax2),
                                    ["Fixed boundaries",
                                     "Periodic boundaries"])):
    if n:
        M[:3,0] = 1
        ax.add_patch(mpl.patches.Rectangle((-.6, -.6), 8.2, 8.2,
                                           linewidth=1, edgecolor='k',
                                           facecolor='none', clip_on=False))
    ax.matshow(M, cmap=CMAP)
    ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False,
        left=False, right=False)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks(np.arange(8)-.5)
    ax.set_yticks(np.arange(8)-.5)
    ax.grid()
    if not n:
        lgd = ax.legend(handles=[mpl.lines.Line2D([0], [0], marker='s', color=(0,0,0,0), label="Agents' neighbours",
                        markerfacecolor=Ngh_CLR, markersize=15)],
                   loc='upper left', bbox_to_anchor=(-0.015, 1.125),
                   frameon=False)
    ax.text(2, 5, '$i$', fontsize=25, **text_params)
    ax.text(7, 1, '$j$', fontsize=25, **text_params)
    ax.set_title(title, y=-0.1)

# plt.figaspect(2)
plt.tight_layout()
plt.savefig('/Users/Luca/Downloads/schelling_grid1.svg', transparent=True,
            bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
