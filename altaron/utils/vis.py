import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

def plot_candlestick(
                    data, 
                    show_volume = True,
                    figsize = (16,6),
                    config = {"plot1" : []}
    ):

    assert (isinstance(data, pd.DataFrame)), "Provide data as a pandas.DataFrame"
    assert (isinstance(show_volume, bool)), "Provide show_true argument as a boolean"
    assert (isinstance(config, dict)), "Provide config as a dict"
    style.use("ggplot")
    if "plot1" not in config:
        config["plot1"] = []

    rows = len(config) + int(show_volume)
    height_ratios = [4]
    for x in range(1, rows):
        height_ratios.append(1)
    fig, axes = plt.subplots(rows, 1, sharex="all", figsize = figsize, gridspec_kw={'height_ratios': height_ratios})
    if not show_volume and (list(config.keys()) == ["plot1"] or config == {}):
        ax_ohlc = axes
    else:
        ax_ohlc = axes[0]

    c = data["Close"].values
    o = data["Open"].values
    l = data["Low"].values
    h = data["High"].values

    for val in config["plot1"]:
        if isinstance(val, list):#For Event Location Specifications
            ax_ohlc.scatter(x=range(len(data)), y=data[val[0]].values, s=175, label=val[0])
        else:
            ax_ohlc.plot(data[val].values, label=val)
    
    colors = np.where(c > o, "green", "red")
    ax_ohlc.bar(x=range(len(c)), height=c - o, 
    bottom=o, width=1, color=colors, edgecolor="black")
    ax_ohlc.vlines(x=range(len(c)), ymin=l, ymax=h, colors=colors)

    
    
    if len(config["plot1"]) > 0:
        ax_ohlc.legend(loc="best")

    if show_volume:
        ax_vol = axes[-1]
        v = data["Volume"].values

        for ind, ax in enumerate(axes[1:-1]):
            for val in config[f"plot{ind+2}"]:
                if isinstance(val, list):
                    ax.scatter(x=range(len(data)), y=data[val[0]].values, s=175, label=val[0])
                else:
                    ax.plot(data[val].values, label=val)
            ax.legend(loc="best")
        
        ax_vol.bar(x=range(len(v)), height=v, color=colors)
    
    else:
        for ind, ax in enumerate(axes[1:]):
            for val in config[f"plot{ind+2}"]:
                if isinstance(val, list):
                    ax.scatter(x=range(len(data)), y=data[val[0]].values, s=175, label=val[0])
                else:
                    ax.plot(data[val].values, label=val)
            ax.legend(loc="best")

    plt.show()
    plt.close()

def plot_pairs(
            data,
            pair_names,
            same_scale= True,
            figsize = (16,6),
    ):

    assert (isinstance(data, pd.DataFrame)), "Provide data as a pandas.DataFrame"
    assert (isinstance(pair_names, list) and len(pair_names) == 2), "Provide pair names as a list of length 2"
    assert (isinstance(same_scale, bool)), "Provide same_scale as a boolean"
    
    style.use("ggplot")

    fig, axes = plt.subplots(2, 1, sharex="all", figsize = figsize, gridspec_kw={'height_ratios': [2,2]})
    ax1, ax_spread = axes

    name1, name2 = pair_names
    if same_scale:
        pair1 = data[name1].iloc[1:]/data[name1].iloc[0]
        pair2 = data[name2].iloc[1:]/data[name2].iloc[0]
    else:
        pair1 = data[name1]
        pair2 = data[name2]
    
    ax1.plot(pair1.values, label=name1)
    ax1.plot(pair2.values, label=name2)
    ax1.legend(loc="best")
    ax1.set_title(f"{name1} vs {name2}")

    spread = pair1 - pair2
    ax_spread.plot(spread.values)
    ax_spread.hlines(y=spread.mean(), color="black", xmin=0, xmax=len(spread)-1)
    ax_spread.hlines(y=spread.mean() - 3*spread.std(), color="blue", xmin=0, xmax=len(spread)-1, linestyle="--")
    ax_spread.hlines(y=spread.mean() + 3*spread.std(), color="blue", xmin=0, xmax=len(spread)-1, linestyle="--")
    ax_spread.set_title("Spread")

    plt.show()
    plt.close()