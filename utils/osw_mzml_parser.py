import argparse
import math
import numpy as np
import os
import pyopenms as ms
import time

from .general_utils import calc_bin_idx

class ExtraTraceConsumer():
    def __init__(self, num_traces, trace_length, out_dir, repl_name, decoy):
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.out_dir = out_dir
        self.repl_name = repl_name
        self.decoy = decoy

    def setExperimentalSettings(self, s):
        pass

    def setExpectedSize(self, a, b):
        pass

    def consumeChromatogram(self, c):
        chromatogram_id = c.getNativeID()

        if self.decoy and b'DECOY' not in chromatogram_id:
            return
        elif not self.decoy and b'DECOY' in chromatogram_id:
            return

        if b'Precursor_i0' in chromatogram_id:
            trace_counter = 0
            extra_traces_npy = np.zeros((self.num_traces, self.trace_length))
            for ms2_array in c.getFloatDataArrays():
                name = ms2_array.getName()

                if b'GrpIsotopeCorr' == name or b'GrpIsotopeOverlap' == name:
                    continue

                ms2_array_npy = np.array([i for i in ms2_array])

                if b'logSNScore' == name:
                    ms2_array_npy = np.nan_to_num(ms2_array_npy, posinf=0.0)
                elif b'LibSAngle' == name:
                    ms2_array_npy = np.nan_to_num(
                        ms2_array_npy, nan=np.nanmax(ms2_array_npy))

                extra_traces_npy[trace_counter, :] = ms2_array_npy
                trace_counter+= 1

            seq_and_charge = b'_'.join(chromatogram_id.split(b'_')[-3].split(b'/'))
            npy_filename = b'_'.join([self.repl_name, seq_and_charge, b'Extra'])
            
            np.save(
                os.path.join(self.out_dir, npy_filename.decode()), extra_traces_npy)

    def consumeSpectrum(self, s):
        pass

class RawData2DExtractionConsumer():
    def __init__(
        self,
        mzml_filename,
        num_ms2_scans=32,
        ms1_min_mz=400,
        ms1_max_mz=1200,
        ms2_min_mz=0,
        ms2_max_mz=2000,
        bin_resolution=0.01,
        expected_cycles=2372,
        outdir='.'):
        self.mzml_filename = mzml_filename.split('/')[-1].split('.')[0]
        self.num_ms2_scans = num_ms2_scans
        self.ms1_array = []
        self.ms2_array = [[] for _ in range(self.num_ms2_scans)]
        self.ms1_rt_array = []
        self.ms2_rt_array = [[] for _ in range(self.num_ms2_scans)]
        self.ms1_min_mz = ms1_min_mz
        self.ms1_max_mz = ms1_max_mz
        self.ms2_min_mz = ms2_min_mz
        self.ms2_max_mz = ms2_max_mz
        self.bin_resolution = bin_resolution
        self.ms1_num_bins = int(
            (self.ms1_max_mz - self.ms1_min_mz) / self.bin_resolution)
        self.ms2_num_bins = int(
            (self.ms2_max_mz - self.ms2_min_mz) / self.bin_resolution)
        self.expected_cycles = expected_cycles
        self.curr_cycle = 0
        self.curr_ms2_scan = 0
        self.outdir = outdir

    def setExperimentalSettings(self, s):
        pass

    def setExpectedSize(self, a, b):
        pass

    def consumeChromatogram(self, c):
        pass

    def consumeSpectrum(self, s):
        ms_level = s.getMSLevel()

        if ms_level > 1:
            num_bins = self.ms2_num_bins
            min_mz = self.ms2_min_mz
        else:
            num_bins = self.ms1_num_bins
            min_mz = self.ms1_min_mz

        binned_rt_slice = [0 for _ in range(num_bins)]

        for mz, i in zip(*s.get_peaks()):
            bin_idx = calc_bin_idx(mz, min_mz, num_bins)
            
            binned_rt_slice[bin_idx]+= i
        
        rt = s.getRT()

        if ms_level > 1:
            print(
                f'Appending MS2 Scan {self.curr_ms2_scan} for '
                f'cycle {self.curr_cycle}'
            )
            self.ms2_array[self.curr_ms2_scan].append(binned_rt_slice)
            self.ms2_rt_array[self.curr_ms2_scan].append(rt)
            self.curr_ms2_scan+= 1
        else:
            self.curr_cycle+= 1
            print(f'Appending MS1 Scan for cycle {self.curr_cycle}')
            self.ms1_array.append(binned_rt_slice)
            self.ms1_rt_array.append(rt)
            self.curr_ms2_scan = 0

        if (
            self.curr_cycle == self.expected_cycles and 
            self.curr_ms2_scan == self.num_ms2_scans
        ):
            ms1_filename = f'{self.mzml_filename}_ms1_array'
            ms2_filename = f'{self.mzml_filename}_ms2_array'
            ms1_rt_filename = f'{self.mzml_filename}_ms1_rt_array'
            ms2_rt_filename = f'{self.mzml_filename}_ms2_rt_array'
            self.ms1_array = np.array(self.ms1_array).transpose((1, 0))
            self.ms2_array = np.array(self.ms2_array).transpose((0, 2, 1))
            self.ms1_rt_array = np.array(self.ms1_rt_array)
            self.ms2_rt_array = np.array(self.ms2_rt_array)

            print(f'Saving ms1 array of shape {self.ms1_array.shape}')
            np.save(os.path.join(self.outdir, ms1_filename), self.ms1_array)
            print(f'Saving ms2 array of shape {self.ms2_array.shape}')
            np.save(os.path.join(self.outdir, ms2_filename), self.ms2_array)
            print(f'Saving ms1 rt array of shape {self.ms1_rt_array.shape}')
            np.save(os.path.join(self.outdir, ms1_rt_filename), self.ms1_rt_array)
            print(f'Saving ms2 rt array of shape {self.ms2_rt_array.shape}')
            np.save(os.path.join(self.outdir, ms2_rt_filename), self.ms2_rt_array)

def extract_additional_info_traces(
    mzML_filename,
    num_traces=15,
    trace_length=2372,
    out_dir='OpenSWATHAutoAnnotatedAllXGB',
    decoy=False):
    repl_name = b'_'.join(mzML_filename.split(b'/')[0].split(b'_')[-3:])
    consumer = ExtraTraceConsumer(
        num_traces, trace_length, out_dir, repl_name, decoy)
    ms.MzMLFile().transform(mzML_filename, consumer)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    # parser.add_argument('-num_traces', '--num_traces', type=int, default=15)
    # parser.add_argument(
    #     '-trace_length', '--trace_length', type=int, default=2372)
    # parser.add_argument(
    #     '-out_dir',
    #     '--out_dir',
    #     type=str,
    #     default='OpenSWATHAutoAnnotatedAllXGB')
    # parser.add_argument(
    #     '-decoy', '--decoy', action='store_true', default=False)
    # parser.add_argument(
    #     '-in_folder',
    #     '--in_folder',
    #     type=str,
    #     default='hroest_K120808_Strep0PlasmaBiolRepl1_R01_SW')
    # args = parser.parse_args()

    # args.in_folder = args.in_folder.split(',')

    # print(args)

    # for folder in args.in_folder:
    #     extract_additional_info_traces(
    #         folder.encode() + b'/output.chrom.mzML',
    #         num_traces=args.num_traces,
    #         trace_length=args.trace_length,
    #         out_dir=args.out_dir,
    #         decoy=args.decoy)

    parser.add_argument('-raw_mzml', '--raw_mzml', type=str, default='')

    args = parser.parse_args()

    consumer = RawData2DExtractionConsumer(args.raw_mzml)
    ms.MzMLFile().transform(args.raw_mzml.encode(), consumer)
    
    print('It took {0:0.1f} seconds'.format(time.time() - start))