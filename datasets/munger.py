import csv
import numpy as np
import os

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

def parse_and_label_skyline_exported_chromatograms_lstm(
    chromatograms_filename, annotations, root_dir):
    chromatograms, labels = [], []

    with open(chromatograms_filename) as infile:
        next(infile)
        last_seq, last_repl, last_charge = '', '', ''
        trace_counter = 0
        chromatogram = None
        
        line_counter = 0
        chromatogram_id = 0

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
                        chromatogram_filename = \
                            '_'.join([last_seq, last_repl, last_charge])

                        chromatograms.append(
                            [chromatogram_id,
                            chromatogram_filename])
                        chromatogram_id+= 1

                        if trace_counter < 6:
                            for i in range(6 - trace_counter):
                                chromatogram = np.vstack(
                                    (chromatogram,
                                     np.zeros(
                                         (1, np.max(chromatogram.shape)))))
                        
                        np.save(os.path.join(root_dir, chromatogram_filename), chromatogram.T)

                        assert chromatogram.T.shape[0] == len(labels[-1])

                    chromatogram = ints
                    trace_counter = 1

                    times = np.float_(line[8].split(','))
                    anno = annotations[repl][seq]
                    label = []

                    for time in times:
                        if anno['start'] <= time <= anno['end']:
                            label.append(1)
                        else:
                            label.append(0)

                    labels.append(np.array(label))

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

        chromatogram_filename = \
            '_'.join([last_seq, last_repl, last_charge])

        chromatograms.append([chromatogram_id, chromatogram_filename])
        chromatogram_id+= 1

        if trace_counter < 6:
            for i in range(6 - trace_counter):
                chromatogram = np.vstack(
                    (chromatogram,
                        np.zeros(
                            (1, np.max(chromatogram.shape)))))
        
        np.save(os.path.join(root_dir, chromatogram_filename), chromatogram.T)

    return chromatograms, np.array(labels)

def parse_and_label_skyline_exported_chromatograms_cnn(
    chromatograms_filename,
    annotations,
    root_dir,
    subsection_width,
    step_size,
    positive_percentage):
    chromatograms, labels = [], []
    x_count, y_count = 0, 0

    with open(chromatograms_filename) as infile:
        next(infile)
        last_seq, last_repl, last_charge = '', '', ''
        trace_counter = 0
        chromatogram = None
        subsection_labels = []
        label_matrix = []
        
        line_counter = 0
        chromatogram_id = 0

        bb_start, bb_end = None, None

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
                        chromatogram_filename = \
                            '_'.join([last_seq, last_repl, last_charge])

                        if trace_counter < 6:
                            for i in range(6 - trace_counter):
                                chromatogram = np.vstack(
                                    (chromatogram,
                                     np.zeros(
                                         (1, np.max(chromatogram.shape)))))

                        chromatograms.append(
                            [chromatogram_id,
                             chromatogram_filename,
                             bb_start,
                             bb_end])

                        np.save(
                            os.path.join(root_dir, chromatogram_filename),
                            chromatogram)
                        x_count+= 1
                        chromatogram_id+= 1
                        subsection_labels = []
                        bb_start, bb_end = None, None

                    chromatogram = ints
                    trace_counter = 1

                    times = np.float_(line[8].split(','))
                    anno = annotations[repl][seq]

                    row_labels = []

                    for time in times:
                        if anno['start'] <= time <= anno['end']:
                            row_labels.append(1)
                        else:
                            row_labels.append(0)
                    
                    row_labels = np.array(row_labels)

                    label_idxs = np.where(row_labels == 1)[0]

                    if len(label_idxs) > 0:
                        bb_start, bb_end = label_idxs[0], label_idxs[-1]

                    num_positive = (row_labels == 1).sum()

                    i = 0
                    while i + subsection_width <= times.shape[0]:
                        num_positive_in_subsection = (
                            row_labels[i:i + subsection_width] == 1).sum()

                        if ((num_positive_in_subsection >= (
                                positive_percentage * num_positive)) or
                            (num_positive_in_subsection == subsection_width)):
                            subsection_labels.append(1)
                        else:
                            subsection_labels.append(0)
                        i+= step_size
                    label_matrix.append(np.array(subsection_labels))
                    y_count+= 1
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

        chromatogram_filename = '_'.join([last_seq, last_repl, last_charge])

        if trace_counter < 6:
            for i in range(6 - trace_counter):
                chromatogram = np.vstack(
                    (chromatogram,
                        np.zeros(
                            (1, np.max(chromatogram.shape)))))

        chromatograms.append(
            [chromatogram_id, chromatogram_filename, bb_start, bb_end])

        np.save(os.path.join(root_dir, chromatogram_filename), chromatogram)
        x_count+= 1

        print(x_count, y_count)

        label_matrix = np.array(label_matrix)
        print(label_matrix.shape)

    return chromatograms, label_matrix

def parse_and_label_skyline_exported_chromatogram_subsections_cnn(
    chromatograms_filename,
    annotations,
    root_dir,
    subsection_width,
    step_size,
    positive_percentage):
    chromatograms = []
    x_count, y_count = 0, 0

    with open(chromatograms_filename) as infile:
        next(infile)
        last_seq, last_repl, last_charge = '', '', ''
        trace_counter = 0
        chromatogram = None
        subsection_labels = []
        
        line_counter = 0
        chromatogram_id = 0

        chromatograms = []

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
                        chromatogram_filename_root = \
                            '_'.join([last_seq, last_repl, last_charge])

                        if trace_counter < 6:
                            for i in range(6 - trace_counter):
                                chromatogram = np.vstack(
                                    (chromatogram,
                                     np.zeros(
                                         (1, np.max(chromatogram.shape)))))

                        i = 0
                        j = 0
                        while i + subsection_width <= chromatogram.shape[1]:
                            chromatogram_filename = '_'.join(
                                [chromatogram_filename_root,
                                 str(i),
                                 'to',
                                 str(i + subsection_width - 1)])

                            chromatograms.append(
                                [chromatogram_id,
                                 chromatogram_filename,
                                 subsection_labels[j]])

                            i+= step_size
                            j+= 1

                            np.save(
                                os.path.join(root_dir, chromatogram_filename),
                                chromatogram[:, i:i + subsection_width])
                            x_count+= 1
                        chromatogram_id+= 1
                        subsection_labels = []

                    chromatogram = ints
                    trace_counter = 1

                    times = np.float_(line[8].split(','))
                    anno = annotations[repl][seq]

                    row_labels = []

                    for time in times:
                        if anno['start'] <= time <= anno['end']:
                            row_labels.append(1)
                        else:
                            row_labels.append(0)
                    
                    row_labels = np.array(row_labels)

                    num_positive = (row_labels == 1).sum()

                    i = 0
                    while i + subsection_width <= times.shape[0]:
                        num_positive_in_subsection = (
                            row_labels[i:i + subsection_width] == 1).sum()

                        if (num_positive_in_subsection >= \
                            positive_percentage * num_positive):
                            subsection_labels.append(1)
                        else:
                            subsection_labels.append(0)
                        i+= step_size
                        y_count+= 1

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

        chromatogram_filename_root = \
            '_'.join([last_seq, last_repl, last_charge])

        if trace_counter < 6:
            for i in range(6 - trace_counter):
                chromatogram = np.vstack(
                    (chromatogram,
                        np.zeros(
                            (1, np.max(chromatogram.shape)))))

        i = 0
        j = 0
        while i + subsection_width <= chromatogram.shape[1]:
            chromatogram_filename = '_'.join(
                [chromatogram_filename_root,
                 str(i),
                 'to',
                 str(i + subsection_width - 1)])

            chromatograms.append(
                [chromatogram_id,
                 chromatogram_filename,
                 subsection_labels[j]])

            i+= step_size
            j+= 1

            np.save(
                os.path.join(root_dir, chromatogram_filename),
                chromatogram[:, i:i + subsection_width])
            x_count+= 1

        print(x_count, y_count)

    return chromatograms

def parse_and_label_skyline_exported_chromatogram_subsections_in_memory_cnn(
    chromatograms_filename,
    annotations,
    root_dir,
    subsection_width,
    step_size,
    positive_percentage):
    chromatograms, full_chromatograms = [], []
    bb_start, bb_end = None, None
    x_count, y_count = 0, 0

    with open(chromatograms_filename) as infile:
        next(infile)
        last_seq, last_repl, last_charge = '', '', ''
        trace_counter = 0
        chromatogram = None
        subsection_labels = []
        
        line_counter = 0
        chromatogram_id = 0
        subsection_counter = 0

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
                        chromatogram_filename = \
                            '_'.join([last_seq, last_repl, last_charge])

                        if trace_counter < 6:
                            for i in range(6 - trace_counter):
                                chromatogram = np.vstack(
                                    (chromatogram,
                                     np.zeros(
                                         (1, np.max(chromatogram.shape)))))

                        full_chromatograms.append(chromatogram)

                        i = 0
                        j = 0
                        while i + subsection_width <= chromatogram.shape[1]:
                            chromatograms.append(
                                [subsection_counter,
                                 chromatogram_id,
                                 i,
                                 i + subsection_width,
                                 subsection_labels[j],
                                 bb_start,
                                 bb_end])

                            i+= step_size
                            j+= 1
                            x_count+= 1
                            subsection_counter+= 1
                        chromatogram_id+= 1
                        subsection_labels = []
                        bb_start, bb_end = None, None

                    chromatogram = ints
                    trace_counter = 1

                    times = np.float_(line[8].split(','))
                    anno = annotations[repl][seq]

                    row_labels = []

                    for time in times:
                        if anno['start'] <= time <= anno['end']:
                            row_labels.append(1)
                        else:
                            row_labels.append(0)
                    
                    row_labels = np.array(row_labels)

                    label_idxs = np.where(row_labels == 1)[0]
                    
                    if len(label_idxs) > 0:
                        bb_start, bb_end = label_idxs[0], label_idxs[-1]

                    num_positive = (row_labels == 1).sum()

                    i = 0
                    while i + subsection_width <= times.shape[0]:
                        num_positive_in_subsection = (
                            row_labels[i:i + subsection_width] == 1).sum()

                        if (num_positive_in_subsection >= \
                            positive_percentage * num_positive):
                            subsection_labels.append(1)
                        else:
                            subsection_labels.append(0)
                        i+= step_size
                        y_count+= 1

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

        chromatogram_filename = \
            '_'.join([last_seq, last_repl, last_charge])

        if trace_counter < 6:
            for i in range(6 - trace_counter):
                chromatogram = np.vstack(
                    (chromatogram,
                        np.zeros(
                            (1, np.max(chromatogram.shape)))))

        full_chromatograms.append(chromatogram)

        i = 0
        j = 0
        while i + subsection_width <= chromatogram.shape[1]:
            chromatograms.append(
                [subsection_counter,
                 chromatogram_id,
                 i,
                 i + subsection_width,
                 subsection_labels[j],
                 bb_start,
                 bb_end])

            i+= step_size
            j+= 1
            x_count+= 1
            subsection_counter+= 1

        print(x_count, y_count)

    chromatograms_ndarray = np.empty(len(full_chromatograms), dtype=np.ndarray)
    for i in range(len(full_chromatograms)):
        chromatograms_ndarray[i] = full_chromatograms[i]

    return chromatograms, chromatograms_ndarray
                

if __name__ == "__main__":
    # For 1DCNN Subsections
    # chromatograms = \
    #     parse_and_label_skyline_exported_chromatogram_subsections_cnn(
    #         '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.tsv',
    #         parse_skyline_exported_annotations(
    #             '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv'),
    #         '../../../data/working/ManualValidationSliced_20_1',
    #         20,
    #         1,
    #         1.0
    # )

    # with open('../../../data/working/ManualValidationSliced_20_1/chromatograms.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['ID', 'Filename', 'Label'])
    #     writer.writerows(chromatograms)

    # For 1DCNN Whole Chromatograms
    chromatograms, labels = \
        parse_and_label_skyline_exported_chromatograms_cnn(
            '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.tsv',
            parse_skyline_exported_annotations(
                '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv'),
            '../../../data/working/ManualValidation',
            30,
            1,
            0.7)

    with open('../../../data/working/ManualValidation/chromatograms.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Filename', 'BB Start', 'BB End'])
        writer.writerows(chromatograms)

    np.save('../../../data/working/ManualValidation/skyline_exported_labels', labels)

    # For 1DCNN Chromatogram Subsections In Memory
    # chromatograms, chromatograms_ndarray = \
    #     parse_and_label_skyline_exported_chromatogram_subsections_in_memory_cnn(
    #         '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.tsv',
    #         parse_skyline_exported_annotations(
    #             '../../../data/raw/SherlockChromes/ManualValidation/SkylineResult500Peptides.csv'),
    #         '../../../data/working/ManualValidationSliced_30_1',
    #         30,
    #         1,
    #         0.7
    # )
    
    # with open('../../../data/working/ManualValidationSliced_30_1/chromatograms.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Subsection ID', 'ID', 'Start', 'End', 'Label', 'BB Start', 'BB End'])
    #     writer.writerows(chromatograms)

    # np.save('../../../data/working/ManualValidationSliced_20_1_InMemory/chromatograms', chromatograms_ndarray)