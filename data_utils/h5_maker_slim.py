#pylint disable=unexpected-keyword-arg
import uproot
import numpy as np
import awkward as ak
import os, glob, h5py, pathlib

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument('--jet_type', type=str, required=True)
_parser.add_argument('--dataset_type', type=str, choices=['train', 'val', 'test'], required=True)
_args = _parser.parse_args()
jet_type = _args.jet_type
dataset_type = _args.dataset_type


def zero_pad_to_numpy_array(awkward_array, target_length=100, pad_value=0):
    padded_array = ak.pad_none(awkward_array, target_length, axis=1, clip=True)
    padded_array = padded_array[ak.num(padded_array, axis=1) <= target_length]
    numpy_array = ak.fill_none(padded_array, pad_value).to_numpy()
    return numpy_array


def make_h5(rfile='test.root'):
    fname = os.path.basename(rfile)
    print(f"[{jet_type}/{dataset_type}] Starting: {fname}", flush=True)
    out_dir = f'/ceph/abal/JetClass/{dataset_type}/{jet_type}'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    jet_feature_names = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
    particle_feature_names = ['eta', 'phi', 'pt']
    dt = h5py.special_dtype(vlen=str)

    with uproot.open(rfile) as root_file:
        tree = root_file['tree']
        px = zero_pad_to_numpy_array(tree['part_px'].array(), 100)
        py = zero_pad_to_numpy_array(tree['part_py'].array(), 100)
        pz = zero_pad_to_numpy_array(tree['part_pz'].array(), 100)
        part_deta = zero_pad_to_numpy_array(tree['part_deta'].array(), 100)
        part_dphi = zero_pad_to_numpy_array(tree['part_dphi'].array(), 100)
        part_pt = np.sqrt(px**2 + py**2)
        part_d0val = zero_pad_to_numpy_array(tree['part_d0val'].array(), 100, pad_value=-999)
        part_dzval = zero_pad_to_numpy_array(tree['part_dzval'].array(), 100, pad_value=-999)
        part_d0err = zero_pad_to_numpy_array(tree['part_d0err'].array(), 100, pad_value=-999)
        part_dzerr = zero_pad_to_numpy_array(tree['part_dzerr'].array(), 100, pad_value=-999)
        part_charge = zero_pad_to_numpy_array(tree['part_charge'].array(), 100, pad_value=-999)
        part_isChargedHadron = zero_pad_to_numpy_array(tree['part_isChargedHadron'].array(), 100, pad_value=-999)
        part_isPhoton = zero_pad_to_numpy_array(tree['part_isPhoton'].array(), 100, pad_value=-999)
        part_isNeutralHadron = zero_pad_to_numpy_array(tree['part_isNeutralHadron'].array(), 100, pad_value=-999)
        part_isElectron = zero_pad_to_numpy_array(tree['part_isElectron'].array(), 100, pad_value=-999)
        part_isMuon = zero_pad_to_numpy_array(tree['part_isMuon'].array(), 100, pad_value=-999)
        E = zero_pad_to_numpy_array(tree['part_energy'].array(), 100)
        jet_pt = tree['jet_pt'].array().to_numpy()
        jet_eta = tree['jet_eta'].array().to_numpy()
        jet_phi = tree['jet_phi'].array().to_numpy()
        jet_energy = tree['jet_energy'].array().to_numpy()
        jet_nparticles = tree['jet_nparticles'].array().to_numpy()
        jet_sdmass = tree['jet_sdmass'].array().to_numpy()
        jet_tau1 = tree['jet_tau1'].array().to_numpy()
        jet_tau2 = tree['jet_tau2'].array().to_numpy()
        jet_tau3 = tree['jet_tau3'].array().to_numpy()
        jet_tau4 = tree['jet_tau4'].array().to_numpy()

    jet_pfc = np.stack([part_deta, part_dphi, part_pt], axis=-1)          # (N, 100, 3)
    jet_pfc_4vec = np.stack([px, py, pz, E], axis=-1)                     # (N, 100, 4)
    jet_features = np.stack((jet_pt, jet_eta, jet_phi, jet_energy, jet_nparticles, jet_sdmass, jet_tau1, jet_tau2, jet_tau3, jet_tau4), axis=-1)

    part_extra = np.stack((part_d0val, part_dzval, part_d0err, part_dzerr, part_charge, part_isChargedHadron, part_isPhoton, part_isNeutralHadron, part_isElectron, part_isMuon), axis=-1)
    part_extra_names = ['part_d0val', 'part_dzval', 'part_d0err', 'part_dzerr', 'part_charge', 'part_isChargedHadron', 'part_isPhoton', 'part_isNeutralHadron', 'part_isElectron', 'part_isMuon']

    assert not np.isnan(part_extra).any(), "Nan in part_extra"
    assert not np.isnan(jet_pfc).any(), "Nan in jet_pfc"
    assert not np.isnan(jet_features).any(), "Nan in jet_features"

    out_fname = fname.replace('.root', '.h5')
    out_file = os.path.join(out_dir, out_fname)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('jetConstituentsList', data=jet_pfc)
        f.create_dataset('jetConstituentsListFourVectors', data=jet_pfc_4vec)
        f.create_dataset('jetFeatures', data=jet_features)
        f.create_dataset('jetFeatureNames', data=np.array(jet_feature_names, dtype=dt))
        f.create_dataset('particleFeatureNames', data=np.array(particle_feature_names, dtype=dt))
        f.create_dataset('jetConstituentsExtra', data=part_extra)
        f.create_dataset('jetConstituentsExtraNames', data=np.array(part_extra_names, dtype=dt))
    print(f"[{jet_type}/{dataset_type}] Done:     {fname}", flush=True)
    return fname


if __name__ == '__main__':
    multi = True
    file_dir = 'train'
    if dataset_type == 'test':
        file_dir = 'test_20M'
    if dataset_type == 'val':
        file_dir = 'val_5M'
    base_path = f'/ceph/bmaier/qml/ae/rawdata/JetClass/{file_dir}'
    file_paths = glob.glob(os.path.join(base_path, f'{jet_type}*.root'))
    n_total = len(file_paths)
    log_interval = 20
    if multi:
        from multiprocessing import Pool
        from time import sleep
        num_cores = min(n_total, 20)
        print(f"[{jet_type}/{dataset_type}] {n_total} files, {num_cores} cores")
        pool = Pool(num_cores); sleep(3)
        for i, _ in enumerate(pool.imap_unordered(make_h5, file_paths), start=1):
            if i % log_interval == 0 or i == n_total:
                print(f"[{jet_type}/{dataset_type}] Progress: {i}/{n_total} files complete", flush=True)
        pool.close(); pool.join()
    else:
        for i, filename in enumerate(file_paths, start=1):
            make_h5(filename)
            if i % log_interval == 0 or i == n_total:
                print(f"[{jet_type}/{dataset_type}] Progress: {i}/{n_total} files complete", flush=True)
