def plot_red_lines(ax, alpha=1):
    for xx in range(7):
        for x in range(9 - xx):
            X = x / (8 - xx) + .005
            if X<1:
                ax.plot([X, X],
                        [0, 10 / (xx + 1)], color='red', linewidth=1, alpha=alpha)
