import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans serif'
plt.rcParams['font.serif'] = ['Computer Modern'] + plt.rcParams['font.serif']

# Set the style of the major and minor grid lines, filled blocks
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

# Set the tick direction
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Set the size of the tick labels.
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# Set the line width of the plot
plt.rcParams['lines.linewidth'] = 2

# Set the figure size
plt.rcParams['figure.figsize'] = [10, 6]

# Set the dpi of the figure
plt.rcParams['figure.dpi'] = 300

# Set the axes label size
plt.rcParams['axes.labelsize'] = 14

# Set the legend font size
plt.rcParams['legend.fontsize'] = 12

# Set the title size
plt.rcParams['axes.titlesize'] = 10

# Set the error bar cap size
plt.rcParams['errorbar.capsize'] = 3

# Set the marker size
plt.rcParams['lines.markersize'] = 8
