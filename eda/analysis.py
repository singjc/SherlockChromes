import math
import numpy as np

from matplotlib import pyplot as plt

def create_histo(data, bins='auto', title="Example data"):
    plt.hist(data, bins=20, alpha=0.5)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('count')

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
        30,
        'Peak Widths'
    )
