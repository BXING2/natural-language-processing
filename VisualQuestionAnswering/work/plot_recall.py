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
    
    x_max = 11
    ax.axis([x_min - x_edge, x_max + x_edge, y_min - y_edge, y_max + y_edge])

    return ax


# 400K: s
# 600-800K: us
# 1000K-1600K: ns
# 1800K-2600K: ps

# variables
ref_recall = [0.119, 0.167, 0.205, 0.226, 0.236, 0.249, 0.253, 0.26,  0.267, 0.271]
beam_recall = [0.22, 0.304, 0.354, 0.396, 0.424, 0.444, 0.459, 0.467, 0.479, 0.487]

# create figures
fig = plt.figure(dpi = 150, figsize = (4.0, 3.0))
left, bottom, width, height = 0.2, 0.15, 0.75, 0.75
ax = fig.add_axes([left, bottom, width, height])

# plot
x = range(1, len(ref_recall)+1)
x = np.array(x)
width = 0.40
x1 = x - 0.5*width
x2 = x + 0.5*width

ax.bar(x1, ref_recall, width=width, color="orange", edgecolor="black")
ax.bar(x2, beam_recall, width=width, color="skyblue", edgecolor="black")


# ticks settings

ax.xaxis.set_minor_locator(ticker.MultipleLocator(5/5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2/5))

ax.tick_params(direction = "out", length = 6, right = False, top = False, which = "major")
ax.tick_params(direction = "out", length = 4, right = False, top = False, which = "minor")


activate_axis_settings = True

# axis settings
if activate_axis_settings == True:

    settings = {}
    
    settings["x_min"] = 0
    settings["x_max"] = 10
    settings["x_delta"] = 5
    settings["y_min"] = 0
    settings["y_max"] = 0.5 #6000 #60 #300 #0.12
    settings["y_delta"] = 0.2 #2000 #20 #100 #0.04
    
    settings["x_edge"] = 0.00 * (settings["x_max"] - settings["x_min"])
    settings["y_edge"] = 0.00 * (settings["y_max"] - settings["y_min"])
    
    settings["x_axis_labels"] = "N"
    settings["y_axis_labels"] = "Recall@N"

    ax = figure_settings(ax, settings)

# ax.set_aspect("equal")
# ax.grid(which="major", axis="both", color="gray", linestyle="-", alpha=0.1)
# ax.plot([0.0, 4.0], [0.0, 4.0], "k--")

# save figures
fig.savefig("figure.tif", bbox_inches="tight") #transparent=True)

