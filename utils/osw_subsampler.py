import argparse
import sqlite3

def get_precursor_id_from_mod_seq_and_charge(
    con,
    cursor,
    mod_seq,
    charge):
    query = \
        """SELECT precursor.ID 
        FROM PRECURSOR precursor LEFT JOIN PRECURSOR_PEPTIDE_MAPPING mapping
        ON precursor.ID = mapping.PRECURSOR_ID LEFT JOIN PEPTIDE peptide
        ON mapping.PEPTIDE_ID = peptide.ID
        WHERE peptide.MODIFIED_SEQUENCE = '{0}'
        AND precursor.CHARGE = {1}""".format(mod_seq, charge)
    res = cursor.execute(query)
    tmp = res.fetchall()

    assert len(tmp) <= 1

    return tmp

def get_precursor_ids(infile, target_file, idx_file=None):
    if idx_file:
        target_idxs = {}
        with open(idx_file, 'r') as idxs:
            for idx in idxs:
                target_idxs[str(int(float(idx)))] = True

    con = sqlite3.connect(infile)
    cursor = con.cursor()

    processed = {}
    prec_ids = []

    with open(target_file, 'r') as targets:
        next(targets)
        for line in targets:
            line = line.split(',')
            idx, target_filename = line[0], line[1]

            if idx_file:
                if idx not in target_idxs:
                    continue
            
            if target_filename in processed:
                continue

            mod_seq, charge = target_filename.split('_')[-2:]

            target_prec_id = get_precursor_id_from_mod_seq_and_charge(
                con, cursor, mod_seq, charge)

            if target_prec_id:
                prec_id = target_prec_id[0][0]
                prec_ids.append(prec_id)

            processed[target_filename] = True

    con.close()

    return prec_ids

def check_sqlite_table(con, table):
    table_present = False
    c = con.cursor()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="{}"'.format(table))
    if c.fetchone()[0] == 1:
        table_present = True
    else:
        table_present = False
    c.fetchall()

    return(table_present)

def subsample_osw(infile, outfile, target_file, idx_file=None):
    conn = sqlite3.connect(infile)
    ms1_present = check_sqlite_table(conn, "FEATURE_MS1")
    ms2_present = check_sqlite_table(conn, "FEATURE_MS2")
    transition_present = check_sqlite_table(conn, "FEATURE_TRANSITION")
    conn.close()

    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript("""
        PRAGMA synchronous = OFF;
        ATTACH DATABASE "{}" AS sdb;
        CREATE TABLE RUN AS SELECT * FROM sdb.RUN;
        DETACH DATABASE sdb;
        """.format(infile))
        
    print("Info: Propagated runs of file {} to {}.".format(infile, outfile))

    prec_ids = get_precursor_ids(infile, target_file, idx_file=idx_file)

    script = \
        """ATTACH DATABASE "{}" AS sdb;
        CREATE TABLE PRECURSOR AS 
            SELECT * 
            FROM sdb.PRECURSOR 
            WHERE ID IN
                (""".format(infile)

    for prec_id in prec_ids:
        script+= str(prec_id) + ", "

    script = script[:-2]
    script+= \
        """);
        DETACH DATABASE sdb;"""

    c.executescript(script)

    print("Info: Subsampled precursor info of file {} to {}.".format(
        infile, outfile))

    script = \
        """ATTACH DATABASE "{}" AS sdb;
        CREATE TABLE TRANSITION_PRECURSOR_MAPPING AS 
            SELECT * 
            FROM sdb.TRANSITION_PRECURSOR_MAPPING 
            WHERE PRECURSOR_ID IN
                (""".format(infile)

    for prec_id in prec_ids:
        script+= str(prec_id) + ", "

    script = script[:-2]
    script+= \
        """);
        DETACH DATABASE sdb;"""

    c.executescript(script)

    print("Info: Subsampled transitions-precursors of file {} to {}.".format(
        infile, outfile))

    c.executescript(
        """ATTACH DATABASE "{}" AS sdb;
        CREATE TABLE TRANSITION AS 
            SELECT *
            FROM sdb.TRANSITION
            WHERE sdb.TRANSITION.ID IN
                (SELECT TRANSITION_ID
                FROM TRANSITION_PRECURSOR_MAPPING);
        DETACH DATABASE sdb;""".format(infile))

    print("Info: Subsampled transitions of file {} to {}.".format(
        infile, outfile))

    script = \
        """ATTACH DATABASE "{}" AS sdb;
        CREATE TABLE FEATURE AS 
            SELECT *
            FROM sdb.FEATURE
            WHERE PRECURSOR_ID IN
                (""".format(infile)

    for prec_id in prec_ids:
        script+= str(prec_id) + ", "

    script = script[:-2]
    script+= \
        """);
        DETACH DATABASE sdb;"""

    c.executescript(script)
    
    print("Info: Subsampled generic features of file {} to {}.".format(
        infile, outfile))

    if ms1_present:
        c.executescript(
            """ATTACH DATABASE "{}" AS sdb;
            CREATE TABLE FEATURE_MS1 AS 
                SELECT *
                FROM sdb.FEATURE_MS1
                WHERE sdb.FEATURE_MS1.FEATURE_ID IN
                    (SELECT ID
                    FROM FEATURE);
            DETACH DATABASE sdb;""".format(infile))

        print("Info: Subsampled MS1 features of file {} to {}.".format(
            infile, outfile))

    if ms2_present:
        c.executescript(
            """ATTACH DATABASE "{}" AS sdb;
            CREATE TABLE FEATURE_MS2 AS 
                SELECT *
                FROM sdb.FEATURE_MS2
                WHERE sdb.FEATURE_MS2.FEATURE_ID IN
                    (SELECT ID
                    FROM FEATURE);
            DETACH DATABASE sdb;""".format(infile))

        print("Info: Subsampled MS2 features of file {} to {}.".format(
            infile, outfile))

    if transition_present:
        c.executescript(
            """ATTACH DATABASE "{}" AS sdb;
            CREATE TABLE FEATURE_TRANSITION AS 
                SELECT *
                FROM sdb.FEATURE_TRANSITION
                WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN
                    (SELECT ID
                    FROM FEATURE);
            DETACH DATABASE sdb;""".format(infile))
        
        print("Info: Subsampled transition features of file {} to {}.".format(
            infile, outfile))

    conn.commit()
    conn.close()

    print("Info: OSW file was subsampled.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile', '--infile', type=str, default='merged.osw')
    parser.add_argument(
        '-outfile', '--outfile', type=str, default='manually_validated.osw')
    parser.add_argument(
        '-targets',
        '--targets',
        type=str,
        default='chromatograms_mixed_all.csv')
    parser.add_argument('-idxs', '--idxs', type=str)
    args = parser.parse_args()

    if 'idxs' not in args:
        args.idxs = None

    subsample_osw(args.infile, args.outfile, args.targets, args.idxs)
