import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def figure_settings(ax, settings):

    x_min = settings["x_min"]
    x_max = settings["x_max"]
    y_min = settings["y_min"]
    y_max = settings["y_max"]
    x_delta = settings["x_delta"]
    y_delta = settings["y_delta"]

    x_edge = settings["x_edge"]
    y_edge = settings["y_edge"]

    x_axis_labels = settings["x_axis_labels"]
    y_axis_labels = settings["y_axis_labels"]

    # ticks settings
    xticks = np.round(np.arange(x_min, x_max + 0.1 * x_delta, x_delta), 1)
    yticks = np.round(np.arange(y_min, y_max + 0.1 * y_delta, y_delta), 3)
    
    # ticklabels settings 
    xticklabels = ["{:.0f}".format(xtick) for xtick in xticks]

    #xticklabels = ["0", "2.5x$10^{6}$", "5x$10^{6}$"]
    #xticklabels = ["0", "5x$10^{7}$", "1x$10^{8}$"]
    #xticklabels = ["0", "5x$10^{5}$", "1x$10^{6}$"]
    #xticklabels = ["0", "2.5x$10^{5}$", "5x$10^{5}$", "7.5x$10^{5}$", "1x$10^{6}$"]
    #yticklabels = ["0", "3x$10^{3}$", "6x$10^{3}$", "9x$10^{3}$", "1.2x$10^{4}$"]
    #yticklabels = ["0", "2.5x$10^{4}$", "5x$10^{4}$", "7.5x$10^{4}$", "1x$10^{5}$"]
    #yticklabels = ["0", "5x$10^{4}$", "1x$10^{5}$", "1.5x$10^{5}$"]
    
    #xticklabels = ["0", "5x$10^{4}$", "1x$10^{5}$", "1.5x$10^{5}$", "2x$10^{5}$"]
    yticklabels = ["{:.1f}".format(ytick) for ytick in yticks]
    
    fontsize = 14

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(xticklabels, fontsize = fontsize)
    ax.set_yticklabels(yticklabels, fontsize = fontsize)

    # axis settings
    ax.set_xlabel(x_axis_labels, fontsize = fontsize)
    ax.set_ylabel(y_axis_labels, fontsize = fontsize)

    '''    
    # axis spines
    directions = ["left", "right", "bottom", "top"]
    linewidth = 1.0
    for direction in directions:
        ax.spines[direction].set_linewidth(linewidth)
    '''

    #ax.spines["right"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    
    # y_max = 3100
    ax.axis([x_min - x_edge, x_max + x_edge, y_min - y_edge, y_max + y_edge])

    return ax


# 400K: s
# 600-800K: us
# 1000K-1600K: ns
# 1800K-2600K: ps

# variables
file_path = "train_valid_metric.npy"
loss = np.load(file_path, allow_pickle=True).item()
train_loss, valid_loss = loss["train_loss"], loss["valid_loss"]

# create figures
fig = plt.figure(dpi = 150, figsize = (4.0, 3.0))
left, bottom, width, height = 0.2, 0.15, 0.75, 0.75
ax = fig.add_axes([left, bottom, width, height])

epoches = range(1, len(valid_loss)+1)

# plot

index = 20
ax.plot(epoches[:index], train_loss[:index], color="red")
ax.plot(epoches[:index], valid_loss[:index], color="blue")


# ticks settings

ax.xaxis.set_minor_locator(ticker.MultipleLocator(10/5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1/5))

ax.tick_params(direction = "out", length = 6, right = False, top = False, which = "major")
ax.tick_params(direction = "out", length = 4, right = False, top = False, which = "minor")


activate_axis_settings = True

# axis settings
if activate_axis_settings == True:

    settings = {}
    
    settings["x_min"] = 0
    settings["x_max"] = 20
    settings["x_delta"] = 10
    settings["y_min"] = 0
    settings["y_max"] = 3 #6000 #60 #300 #0.12
    settings["y_delta"] = 1 #2000 #20 #100 #0.04
    
    settings["x_edge"] = 0.00 * (settings["x_max"] - settings["x_min"])
    settings["y_edge"] = 0.00 * (settings["y_max"] - settings["y_min"])
    
    settings["x_axis_labels"] = "Epoch"
    # settings["y_axis_labels"] = "Mean distortion"
    settings["y_axis_labels"] = "Loss"

    ax = figure_settings(ax, settings)

# ax.set_aspect("equal")
# ax.grid(which="major", axis="both", color="gray", linestyle="-", alpha=0.1)
# ax.plot([0.0, 4.0], [0.0, 4.0], "k--")

# save figures
fig.savefig("figure.tif", bbox_inches="tight") #transparent=True)

