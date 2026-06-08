"""
Flatten the jet-pt distribution by sampling a fixed number of jets per pt bin.
Reads h5 files produced by h5_maker_slim.py and writes a single flat-pt sampled h5 per jet type.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import os, glob, h5py, pathlib, argparse
import numpy as np
import tqdm
from sklearn.utils import resample


BINS = np.arange(500., 1001., 50.)   # pt bins in GeV, [500, 550, ..., 1000]


def flatten_feature_distribution(feature: np.ndarray, num_events: int) -> list[int]:
    """Return indices that sample up to num_events jets from each pt bin."""
    bin_indices = np.digitize(feature, BINS)
    sampled = []
    for b in range(1, len(BINS)):
        in_bin = np.where(bin_indices == b)[0]
        if len(in_bin) > 0:
            n = min(len(in_bin), num_events)
            sampled.extend(resample(in_bin, replace=False, n_samples=n))
    return sampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--purpose', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--signal', type=str, required=True)
    parser.add_argument('--num_events_per_bin', type=int, default=100)
    args = parser.parse_args()

    purpose = args.purpose
    signal = args.signal
    num_events_per_bin = args.num_events_per_bin

    base_dir = f'/ceph/abal/JetClass/{purpose}/{signal}/'
    out_dir  = f'/ceph/abal/JetClass/flat_{purpose}/{signal}/'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(base_dir, '*.h5')))
    JET_PT_IDX = 0   # jet_pt is the first column in jetFeatures

    # Read string metadata from first file before the main loop
    with h5py.File(file_paths[0], 'r') as f0:
        jet_feature_names   = f0['jetFeatureNames'][()]
        particle_feat_names = f0['particleFeatureNames'][()]
        extra_feature_names = f0['jetConstituentsExtraNames'][()]

    pfc, pfc_4vec, jet_feats, pfc_extra = [], [], [], []

    for file_path in tqdm.tqdm(file_paths):
        with h5py.File(file_path, 'r') as f:
            idx = flatten_feature_distribution(f['jetFeatures'][:, JET_PT_IDX], num_events_per_bin)
            pfc.append(f['jetConstituentsList'][()][idx])
            pfc_4vec.append(f['jetConstituentsListFourVectors'][()][idx])
            jet_feats.append(f['jetFeatures'][()][idx])
            pfc_extra.append(f['jetConstituentsExtra'][()][idx])

    pfc       = np.concatenate(pfc,      axis=0)
    pfc_4vec  = np.concatenate(pfc_4vec, axis=0)
    jet_feats = np.concatenate(jet_feats, axis=0)
    pfc_extra = np.concatenate(pfc_extra, axis=0)

    # 0 for ZJetsToNuNu (background), 1 for everything else (signal)
    truth_label  = 0 if 'ZJets' in signal else 1
    truth_labels = np.full(jet_feats.shape[0], truth_label, dtype=np.float32)

    # shuffle all arrays together after concatenation (order was file -> pt bin)
    perm     = np.random.default_rng(141098).permutation(len(pfc))
    pfc      = pfc[perm]
    pfc_4vec = pfc_4vec[perm]
    jet_feats    = jet_feats[perm]
    pfc_extra    = pfc_extra[perm]
    truth_labels = truth_labels[perm]

    out_file = os.path.join(out_dir, f'{signal}_flat_pt_sample.h5')
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('jetConstituentsList',            data=pfc)
        f.create_dataset('jetConstituentsListFourVectors', data=pfc_4vec)
        f.create_dataset('jetFeatures',                    data=jet_feats)
        f.create_dataset('jetFeatureNames',                data=jet_feature_names)
        f.create_dataset('particleFeatureNames',           data=particle_feat_names)
        f.create_dataset('jetConstituentsExtra',           data=pfc_extra)
        f.create_dataset('jetConstituentsExtraNames',      data=extra_feature_names)
        f.create_dataset('truth_labels',                   data=truth_labels)
        f.attrs['binning']            = str(BINS.tolist())
        f.attrs['num_events_per_bin'] = num_events_per_bin

    print(f"Saved {jet_feats.shape[0]} events -> {out_file}")
