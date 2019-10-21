import math
import numpy as np

from matplotlib import pyplot as plt

def create_histo(
    data,
    bins='auto',
    title="Example data",
    percentiles=[50, 50],
    labels=['Real', 'Decoy']):
    for i in range(len(data)):
        print(np.percentile(data[i], percentiles[i]))
        plt.hist(data[i], bins=bins, alpha=0.5, label=labels[i])

    plt.title(title)
    plt.xlabel(title)

    lower, upper = int(min(data[0])), int(max(data[0]))

    if upper <= 1:
        xticks = [0.1 * i for i in range(0, 11)]
    else:
        mult = 1
        xticks = [i for i in range(lower - 1, upper + 1)]
        while len(xticks) > 100:
            xticks = [i for i in range(lower - 1, upper + 1, mult)]
            mult*= 10

    plt.xticks(xticks)
    plt.ylabel('count')

    plt.legend(loc='upper right')
    plt.show()

def get_peak_width_dist(annotations_filename):
    peak_widths = []

    with open(annotations_filename) as infile:
        next(infile)
        
        for line in infile:
            line = line.rstrip('\r\n').split(',')
            start, end = line[-2], line[-1]

            if start != '#N/A' and end != '#N/A':
                peak_widths.append(
                    (float(end) - float(start)) / 0.0569)

    return peak_widths


if __name__ == "__main__":
    create_histo(
        get_peak_width_dist(
            '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv'
        ),
        60,
        'Peak Widths'
    )
