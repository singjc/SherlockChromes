import argparse
import bisect
import csv
import numpy as np
import os
import sqlite3
import time

from sql_data_access import SqlDataAccess

def get_run_id_from_folder_name(
    con,
    cursor,
    folder_name):
    query = \
        """SELECT ID FROM RUN WHERE FILENAME LIKE '%{0}%'""".format(
            folder_name)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) == 1

    return tmp[0][0]

def get_mod_seqs_and_charges_from_prec_ids(
    con,
    cursor,
    prec_id_lower=0,
    prec_id_upper=9,
    decoy=0):
    query = \
        """SELECT peptide.MODIFIED_SEQUENCE, precursor.CHARGE 
        FROM PRECURSOR precursor LEFT JOIN PRECURSOR_PEPTIDE_MAPPING mapping
        ON precursor.ID = mapping.PRECURSOR_ID LEFT JOIN PEPTIDE peptide
        ON mapping.PEPTIDE_ID = peptide.ID
        WHERE precursor.ID BETWEEN {0} AND {1} 
        AND precursor.DECOY = {2}""".format(
            prec_id_lower, prec_id_upper, decoy)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) == (prec_id_upper - prec_id_lower + 1), \
        str(prec_id_upper - prec_id_lower + 1) \
        + ' ' \
        + str(len(tmp))

    return tmp

def get_feature_info_from_run_and_precursor_ids(
    con,
    cursor,
    run_id,
    prec_id_lower=0,
    prec_id_upper=9,
    decoy=0):
    query = \
        """SELECT f.EXP_RT, f.DELTA_RT, f.LEFT_WIDTH, f.RIGHT_WIDTH, s.SCORE
        FROM PRECURSOR p
		LEFT JOIN FEATURE f ON p.ID = f.PRECURSOR_ID 
		AND p.DECOY = {0} 
		AND (f.RUN_ID = {1} OR f.RUN_ID IS NULL) 
		LEFT JOIN SCORE_MS2 s ON f.ID = s.FEATURE_ID 
		WHERE p.ID BETWEEN {2} AND {3}
		AND (s.RANK = 1 OR s.RANK IS NULL) 
        ORDER BY p.ID ASC""".format(
            decoy, run_id, prec_id_lower, prec_id_upper)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) == (prec_id_upper - prec_id_lower + 1), \
        str(run_id) \
        + ' ' \
        + str(prec_id_upper - prec_id_lower + 1) \
        + ' ' \
        + str(len(tmp))
    
    return tmp

def get_transition_ids_and_library_intensities_from_prec_id(
    con,
    cursor,
    prec_id,
    decoy=0):
    query = \
        """SELECT ID, LIBRARY_INTENSITY 
        FROM TRANSITION LEFT JOIN TRANSITION_PRECURSOR_MAPPING
        ON TRANSITION.ID = TRANSITION_ID
        WHERE PRECURSOR_ID = {0} AND DECOY = {1}""".format(prec_id, decoy)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) > 0, prec_id
    
    return tmp

def get_ms2_chromatogram_ids_from_transition_ids(con, cursor, transition_ids):
    sql_query = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN ("

    for current_id in transition_ids:
        sql_query+= "'" + current_id + "', "

    sql_query = sql_query[:-2]
    sql_query = sql_query + ') ORDER BY NATIVE_ID ASC'

    res = cursor.execute(sql_query)
    tmp = res.fetchall()

    # assert len(tmp) > 0, str(transition_ids)

    return tmp

def get_ms1_chromatogram_ids_from_precursor_id_and_isotope(
    con,
    cursor,
    prec_id,
    isotopes):
    sql_query = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN ("

    for isotope in isotopes:
        sql_query+= "'{0}_Precursor_i{1}', ".format(prec_id, isotope)

    sql_query = sql_query[:-2]
    sql_query = sql_query + ') ORDER BY NATIVE_ID ASC'

    res = cursor.execute(sql_query)
    tmp = res.fetchall()

    assert len(tmp) > 0, str(prec_id) + ' ' + str(isotope)

    return tmp

def get_chromatogram_labels_and_bbox(
    left_width,
    right_width,
    times):
    row_labels = []

    for time in times:
        if left_width and right_width:
            if left_width <= time <= right_width:
                row_labels.append(1)
            else:
                row_labels.append(0)
        else:
            row_labels.append(0)
    
    row_labels = np.array(row_labels)

    label_idxs = np.where(row_labels == 1)[0]

    if len(label_idxs) > 0:
        bb_start, bb_end = label_idxs[0], label_idxs[-1]
    else:
        bb_start, bb_end = None, None

    return row_labels, bb_start, bb_end

def create_data_from_transition_ids(
    sqMass_dir,
    sqMass_filename,
    transition_ids,
    out_dir,
    chromatogram_filename,
    left_width,
    right_width,
    prec_id=None,
    isotopes=[],
    library_intensities=[],
    exp_rt=None,
    extra_features=[],
    csv_only=False,
    window_size=201):
    con = sqlite3.connect(os.path.join(sqMass_dir, sqMass_filename))

    cursor = con.cursor()

    ms2_transition_ids = get_ms2_chromatogram_ids_from_transition_ids(
        con, cursor, transition_ids)

    if len(ms2_transition_ids) == 0:
        return -1, -1, -1

    ms2_transition_ids = [item[0] for item in ms2_transition_ids]

    transitions = SqlDataAccess(os.path.join(sqMass_dir, sqMass_filename))

    ms2_transitions = transitions.getDataForChromatograms(
        ms2_transition_ids)

    times = ms2_transitions[0][0]
    len_times = len(times)
    subsection_left, subsection_right = 0, len_times

    row_labels, bbox_start, bbox_end = get_chromatogram_labels_and_bbox(
            left_width,
            right_width,
            times)

    if not csv_only:
        num_expected_features = 6
        num_expected_extra_features = 0
        free_idx = 0

        if 'exp_rt' in extra_features:
            num_expected_extra_features+= 1

        if 'lib_int' in extra_features:
            num_expected_extra_features+= 6

        if 'ms1' in extra_features:
            num_expected_extra_features+= len(isotopes)

        chromatogram = np.zeros((num_expected_features, len_times))
        extra = np.zeros((num_expected_extra_features, len_times))

        ms2_transitions = np.array(
            [transition[1] for transition in ms2_transitions])

        assert ms2_transitions.shape[1] > 1, print(chromatogram_filename)

        chromatogram[0:ms2_transitions.shape[0]] = ms2_transitions

        if 'exp_rt' in extra_features:
            dist_from_exp_rt = np.absolute(
                np.repeat(exp_rt, len_times) - np.array(times))

            extra[free_idx:free_idx + 1] = dist_from_exp_rt
            free_idx+= 1

        if 'lib_int' in extra_features:
            lib_int_features = np.repeat(
                library_intensities,
                len_times).reshape(len(library_intensities), len_times)
            
            extra[free_idx:free_idx + lib_int_features.shape[0]] = (
                lib_int_features)
            free_idx+= 6
        
        if 'ms1' in extra_features:
            ms1_transition_ids = \
                get_ms1_chromatogram_ids_from_precursor_id_and_isotope(
                    con, cursor, prec_id, isotopes)

            ms1_transition_ids = [item[0] for item in ms1_transition_ids]

            ms1_transitions = transitions.getDataForChromatograms(
                ms1_transition_ids)

            ms1_transitions = np.array(
                [transition[1] for transition in ms1_transitions])

            extra[free_idx:free_idx + ms1_transitions.shape[0]] = (
                ms1_transitions) 
            free_idx+= len(isotopes)

        if window_size >= 0:
            half_span = window_size // 2
            exp_rt_idx = bisect.bisect(times, exp_rt)
            subsection_left, subsection_right = (
                exp_rt_idx - half_span, exp_rt_idx + half_span + 1)
            if subsection_left < 0:
                subsection_left, subsection_right = 0, window_size
            elif subsection_right >= len_times:
                subsection_left = len_times - window_size

            chromatogram = chromatogram[:, subsection_left:subsection_right]
            extra = extra[:, subsection_left:subsection_right]
            times = times[subsection_left:subsection_right]
            row_labels = row_labels[subsection_left:subsection_right]

            label_idxs = np.where(row_labels == 1)[0]

            if len(label_idxs) > 0:
                bb_start, bb_end = label_idxs[0], label_idxs[-1]
            else:
                bb_start, bb_end = None, None

        np.save(os.path.join(out_dir, chromatogram_filename), chromatogram)
        np.save(
            os.path.join(out_dir, chromatogram_filename + '_Extra'),
            extra)

    return row_labels, bbox_start, bbox_end

def get_cnn_data(
    out_dir,
    osw_dir='.',
    osw_filename='merged.osw',
    sqMass_roots=[],
    prec_id_lower=0,
    prec_id_upper=9,
    decoy=0,
    isotopes=[0],
    extra_features=['exp_rt', 'lib_int', 'ms1'],
    csv_only=False,
    window_size=201,
    use_rt=False,
    scored=False):
    label_matrix, chromatograms_csv = [], []

    chromatogram_id = 0

    con = sqlite3.connect(os.path.join(osw_dir, osw_filename))
    cursor = con.cursor()

    prec_mod_seqs_and_charges = get_mod_seqs_and_charges_from_prec_ids(
            con,
            cursor,
            prec_id_lower,
            prec_id_upper,
            decoy)

    for sqMass_root in sqMass_roots:
        run_id = get_run_id_from_folder_name(con, cursor, sqMass_root)

        if use_rt and scored:
            feature_info = get_feature_info_from_run_and_precursor_ids(
                con,
                cursor,
                run_id,
                prec_id_lower,
                prec_id_upper,
                decoy)

        for prec_id in range(prec_id_lower, prec_id_upper + 1):
            print(prec_id)

            prec_mod_seq, prec_charge = (
                prec_mod_seqs_and_charges[prec_id - prec_id_lower][:])

            transition_ids_and_library_intensities = \
                get_transition_ids_and_library_intensities_from_prec_id(
                    con,
                    cursor,
                    prec_id,
                    decoy)
            transition_ids = \
                [str(x[0]) for x in transition_ids_and_library_intensities]
            library_intensities = \
                [x[1] for x in transition_ids_and_library_intensities]

            if use_rt and scored:
                exp_rt, delta_rt, left_width, right_width, score = \
                feature_info[prec_id - prec_id_lower]

                if exp_rt and delta_rt:
                    exp_rt = exp_rt - delta_rt
                else:
                    continue
            else:
                assert window_size == -1, print(
                    'Cannot subset without using library RT!')

            if not scored:
                # TODO: Implement extraction of OSW features only
                exp_rt, left_width, right_width, score = -1, -1, -1, -1 

            repl_name = sqMass_root
            
            chromatogram_filename = [repl_name, prec_mod_seq, str(prec_charge)]
            if decoy == 1:
                chromatogram_filename.insert(0, 'DECOY')

            chromatogram_filename = '_'.join(chromatogram_filename)

            labels, bbox_start, bbox_end = create_data_from_transition_ids(
                sqMass_root,
                'output.sqMass',
                transition_ids,
                out_dir,
                chromatogram_filename,
                left_width,
                right_width,
                prec_id=prec_id,
                isotopes=isotopes,
                library_intensities=library_intensities,
                exp_rt=exp_rt,
                extra_features=extra_features,
                csv_only=csv_only,
                window_size=window_size)

            if not isinstance(labels, np.ndarray) and labels == -1:
                continue

            if not csv_only:
                label_matrix.append(labels)

            chromatograms_csv.append(
                [
                    chromatogram_id,
                    chromatogram_filename,
                    bbox_start,
                    bbox_end,
                    score,
                    exp_rt,
                    window_size
                ])
            chromatogram_id+= 1

    con.close()

    if decoy == 0:
        npy_filename = 'osw_point_labels'
        csv_filename = 'chromatograms.csv'
    else:
        npy_filename = 'osw_point_labels_decoy'
        csv_filename = 'chromatograms_decoy.csv'

    if not csv_only and scored:
        np.save(os.path.join(
            out_dir, npy_filename), np.array(label_matrix))

    with open(os.path.join(out_dir, csv_filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'ID', 'Filename', 'BB Start', 'BB End', 'OSW Score', 'Lib RT',
                'Window Size'
        ])
        writer.writerows(chromatograms_csv)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, default='osw_parser_out')
    parser.add_argument('-osw_dir', '--osw_dir', type=str, default='.')
    parser.add_argument('-osw_in', '--osw_in', type=str, default='merged.osw')
    parser.add_argument(
        '-in_folder',
        '--in_folder',
        type=str,
        default='hroest_K120808_Strep0PlasmaBiolRepl1_R01_SW')
    parser.add_argument(
        '-prec_id_lower', '--prec_id_lower', type=int, default=0)
    parser.add_argument(
        '-prec_id_upper', '--prec_id_upper', type=int, default=9)
    parser.add_argument('-decoy', '--decoy', type=int, default=0)
    parser.add_argument('-isotopes', '--isotopes', type=str, default='0')
    parser.add_argument(
        '-extra_features',
        '--extra_features',
        type=str,
        default='exp_rt,lib_int,ms1')
    parser.add_argument(
        '-csv_only',
        '--csv_only',
        action='store_true',
        default=False)
    parser.add_argument('-window_size', '--window_size', type=int, default=201)
    parser.add_argument(
        '-use_rt',
        '--use_rt',
        action='store_true',
        default=False)
    parser.add_argument(
        '-scored',
        '--scored',
        action='store_true',
        default=False)
    args = parser.parse_args()

    args.in_folder = args.in_folder.split(',')
    args.isotopes = args.isotopes.split(',')
    args.extra_features = args.extra_features.split(',')

    print(args)

    get_cnn_data(
        out_dir=args.out_dir,
        osw_dir=args.osw_dir,
        osw_filename=args.osw_in,
        sqMass_roots=args.in_folder,
        prec_id_lower=args.prec_id_lower,
        prec_id_upper=args.prec_id_upper,
        decoy=args.decoy,
        isotopes=args.isotopes,
        extra_features=args.extra_features,
        csv_only=args.csv_only,
        window_size=args.window_size,
        use_rt=args.use_rt,
        scored=args.scored)

    print('It took {0:0.1f} seconds'.format(time.time() - start))
