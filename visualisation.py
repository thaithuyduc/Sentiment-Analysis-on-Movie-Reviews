import math
import itertools
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

class LinePlot:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    @staticmethod
    def readable_numbers(x: float) -> str:
        """
        takes a large number and formats it into K,M to make it more readable

        Args
            x: float value to format

        Returns
            str: formatted string
        """
        if x >= 1e6:
            s = '{:1.2f}M'.format(x*1e-6)
        else:
            s = '{:1.2f}K'.format(x*1e-3)
        return s
    
    def set_locator(self, axes: Axes):
        """
        Format x, y ticklabels into more readable form

        Args
            axes: matplotlib axes to format x, y ticklabels
        """
        def is_number(x):
            x = x.replace('−', '-')
            try:
                float(x)
                return True
            except ValueError:
                return False
    
        x_ticklabels = [tick.get_text() for tick in axes.get_xticklabels()]
        y_ticklabels = [tick.get_text() for tick in axes.get_yticklabels()]
    
        formatter = FuncFormatter(lambda val, pos: self.readable_numbers(val))

        if is_number(x_ticklabels[0]) == True:
            axes.xaxis.set_major_formatter(formatter)
        if is_number(y_ticklabels[0]) == True:
            axes.yaxis.set_major_formatter(formatter)
    
    def single_plot(
        self,
        x_axis: str,
        y_axis: str,
        hue: str | None = None,
        style: str | None = None,
        size: str | None = None,
        kwarg: dict | None = None,
        readable_label: bool = False,
        ax: Axes = None
    ) -> Axes:
        """
        Draw a visually stunning and readable line plot.
    
        Args:
            x_axis: categorical or numeric column for X-axis
            y_axis: numeric column for Y-axis
            hue: column for line colors (group)
            style: column for line styles (e.g., dashed, dotted)
            size: column for line thickness variation
            kwarg: additional seaborn.lineplot arguments
            readable_label: whether to format axes tick labels into readable form
            ax: matplotlib Axes to draw plot on
        """
    
        if ax is None:
            ax = plt.gca()
            fig = ax.get_figure()
            fig.set_size_inches(10, 6)
        if kwarg is None:
            kwarg = {}
    
        # --- 🎨 Line Plot ---
        line = sns.lineplot(
            data=self.df,
            x=x_axis,
            y=y_axis,
            hue=hue,
            style=style,
            size=size,
            ax=ax,
            # marker="o",             # ✅ Add circle markers to each data point
            linewidth=2.2,          # ✅ Slightly thicker lines
            # markersize=4,           # ✅ Larger markers for visibility
            alpha=0.9,              # ✅ Soft transparency for overlapping lines
            palette="viridis",      # ✅ Modern, readable color palette
            **kwarg
        )
    
        # --- 🎭 Title & Axis Labels ---
        ax.set_title(
            f"{y_axis} vs {x_axis}",
            fontsize=12,
            fontweight="regular",
            # fontname="Times New Roman",
            # pad=15
        )
        ax.set_xlabel(x_axis, fontsize=12, labelpad=5,) # fontname="Times New Roman"
        ax.set_ylabel(y_axis, fontsize=12, labelpad=10,) # fontname="Times New Roman"
    
        # --- 📏 Tick Formatting ---
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        ax.tick_params(axis="y", rotation=0, labelsize=11)
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    
        # --- 🎨 Legend ---
        if hue or style or size:
            leg = ax.legend(
                title="Legend",
                title_fontsize=11,
                fontsize=10,
                loc="best",
                frameon=False,
                fancybox=True,
                framealpha=0.9,
            )
            leg.get_frame().set_linewidth(0.7)
            leg.get_frame().set_edgecolor("gray")
    
        # --- 📊 Optional: Format Large Axis Labels ---
        if readable_label:
            self.set_locator(ax)
    
        # --- 💅 Aesthetic Cleanup ---
        sns.despine(ax=ax, top=True, right=True)
    
        return line

    def multiple_plots(self, x_axis_list: list[str], y_axis_list: list[str], hue_list: list[str|None], 
                       kwarg: dict, plots_each_row: int, size_list: list[str|None], style_list: list[str|None],
                       readable_label: bool = False) -> tuple[Figure, np.array[Axes]]:
        """
        Draw multiple plots

        Args
            Same as single_plot
            y_axis_list: list of columns for y axis
            x_axis_list: list of columns for x axis
            hue_list
            size_list
            style_list
            plots_each_row: number of plots for each row in matplotlib figure
            kwarg: additional seaborn.lineplot arguments
        """
        
        combination = list(zip(x_axis_list, y_axis_list, hue_list, size_list, style_list))
        
        length = len(combination)
        number_rows = math.ceil(length / plots_each_row)
    
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 10))
        axes = np.array(axes).reshape(number_rows, plots_each_row)
    
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
    
        for coor, com in zip(coordinates, combination):
            x, y, hue, size, style = com
            ax = axes[coor]
            self.single_plot(
                x_axis=x,
                y_axis=y,
                hue=hue,
                size=size,
                style=style,
                readable_label=readable_label,
                kwarg=kwarg,
                ax=ax
            )
    
        # hide any unused axes
        for coor in coordinates[length:]:
            fig.delaxes(axes[coor])

        return fig, axes