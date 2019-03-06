import numpy as np

def parse_skyline_exported_annotations(annotations_filename):
        annotations = {}

        with open(annotations_filename) as infile:
            next(infile)

            for line in infile:
                line = line.rstrip('\r\n').split(',')

                repl, seq, start, end = line[2], line[13], line[15], line[16]

                if repl not in annotations:
                    annotations[repl] = {}

                if seq not in annotations[repl]:
                    if start == '#N/A' or end == '#N/A':
                        start, end = 0.0, 0.0

                    annotations[repl][seq] = (
                        {'start': float(start), 'end': float(end)})

        return annotations

def parse_skyline_exported_chromatogram_filenames(filename):
    filename = filename.split('%')
    repl = filename[0].split('_')[2] + filename[1].split('.')[0][10:]

    return repl

def parse_and_label_skyline_exported_chromatograms(
    chromatograms_filename, annotations):
    chromatograms, labels = [], []

    with open(chromatograms_filename) as infile:
        next(infile)
        counter = 0
        chromatogram = []

        for line in infile:
            line = line.rstrip('\r\n').split('\t')

            if line[1] != '#N/A':
                repl = parse_skyline_exported_chromatogram_filenames(line[0])
                seq, ints = (
                    line[1],
                    np.float_(line[9].split(',')))

                if len(chromatogram) == 0:
                    chromatogram = ints
                else:
                    np.concatenate((chromatogram, ints))

                if counter == 5:
                    chromatograms.append(chromatogram.T)
                    chromatogram = []

                    times = np.float_(line[8].split(','))
                    annotation = annotations[repl][seq]
                    label = []

                    for time in times:
                        if annotation['start'] <= time <= annotation['end']:
                            label.append(1)
                        else:
                            label.append(0)

                    labels.append(np.array(label))

                    counter = 0
                else:
                    counter+= 1

    return np.array(chromatograms), np.array(labels)
                

if __name__ == "__main__":
    chromatograms, labels = parse_and_label_skyline_exported_chromatograms(
        '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.tsv',
        parse_skyline_exported_annotations(
            '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv')
    )

    np.save('../../../data/working/skyline_exported_chromatograms', chromatograms)
    np.save('../../../data/working/skyline_exported_labels', labels)