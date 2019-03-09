import csv
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
    chromatogram_info, chromatograms, labels = [], [], []

    with open(chromatograms_filename) as infile:
        next(infile)
        last_seq, last_repl, last_charge = '', '', ''
        trace_counter = 0
        chromatogram = None
        
        line_counter = 0

        for line in infile:
            line_counter+= 1
            line = line.rstrip('\r\n').split('\t')

            if line[1] != '#N/A':
                repl = parse_skyline_exported_chromatogram_filenames(line[0])
                seq, ints, charge = (
                    line[1],
                    np.float_(line[9].split(',')),
                    line[2])

                if seq != last_seq \
                    or charge != last_charge \
                    or trace_counter == 6:
                    if chromatogram is not None:
                        chromatogram_info.append(
                            [last_seq, last_repl, last_charge])

                        if trace_counter < 6:
                            for i in range(6 - trace_counter):
                                chromatogram = np.vstack(
                                    (chromatogram,
                                     np.zeros(
                                         (1, np.max(chromatogram.shape)))))
                        
                        chromatograms.append(chromatogram.T)

                        times = np.float_(line[8].split(','))
                        anno = annotations[last_repl][last_seq]
                        label = []

                        for time in times:
                            if anno['start'] <= time <= anno['end']:
                                label.append(1)
                            else:
                                label.append(0)

                        labels.append(np.array(label))

                    chromatogram = ints
                    trace_counter = 1

                    last_seq, last_repl, last_charge = seq, repl, charge
                else:
                    chromatogram = np.vstack((chromatogram, ints))
                    trace_counter+= 1
                    
                print(
                    line_counter,
                    seq,
                    charge,
                    chromatogram.shape,
                    ints.shape,
                    trace_counter)

    return chromatogram_info, np.array(chromatograms), np.array(labels)
                

if __name__ == "__main__":
    chromatogram_info, chromatograms, labels = \
        parse_and_label_skyline_exported_chromatograms(
            '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.tsv',
            parse_skyline_exported_annotations(
                '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv')
    )

    with open('../../../data/working/chromatogram_info.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Modified Sequence', 'Replicate', 'Precursor Charge'])
        writer.writerows(chromatogram_info)

    np.save('../../../data/working/skyline_exported_chromatograms', chromatograms)
    np.save('../../../data/working/skyline_exported_labels', labels)
    