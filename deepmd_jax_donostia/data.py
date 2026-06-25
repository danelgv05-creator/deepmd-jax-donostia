import numpy as np
import jax.numpy as jnp
from jax import vmap
from glob import glob
from os.path import abspath
from ase.io import read
from .utils import shift, get_relative_coord, sr


def _classify_path(p):
    return 'extxyz' if isinstance(p, str) and p.lower().endswith(('.xyz', '.extxyz')) else 'dp'


def _flatten_paths(paths):
    for p in paths:
        if isinstance(p, list):
            yield from _flatten_paths(p)
        else:
            yield p


def Dataset(paths, labels, params=None, chemical_types=None):
    """
    Create a dataset object from file paths.

    This function dispatches to the appropriate dataset class based on the input paths
    and formats. It supports DeepMD (DP) format (.npy files in directories) and
    extended XYZ format (.xyz/.extxyz files).

    Parameters
    ----------
    paths : str or list of str
        File paths to load data from. For DP format, paths should be strings pointing
        to directories containing 'type.raw' and 'set.*/' subdirectories with .npy files.
        For extxyz format, paths should be strings ending in '.xyz' or '.extxyz'.
        If paths is a list, it creates a composite dataset from multiple subsets.
    labels : list of str
        List of data labels to load, e.g., ['coord', 'energy', 'force'].
    params : dict, optional
        Additional parameters for dataset configuration, such as 'atomic_sel'.
    chemical_types : list of int, optional
        List of atomic numbers defining the chemical types. If None, inferred from data.

    Returns
    -------
    DatasetLeaf, DatasetGroup, ExtXYZDataset, or DPDataset
        - DatasetLeaf: For single composition groups.
        - DatasetGroup: For composite datasets with multiple subsets.
        - ExtXYZDataset: For extxyz files, groups frames by composition into DatasetLeaf subsets.
        - DPDataset: For a DP format directory.

    Constraints:
    - Mixing DP directories and extxyz files in the same paths list is not supported.
    """
    flat_paths = [paths] if isinstance(paths, str) else list(_flatten_paths(paths))
    if len(flat_paths) == 1:
        path = flat_paths[0]
        if _classify_path(path) == 'extxyz':
            return ExtXYZDataset([path], labels, params, chemical_types)
        return DPDataset(path, labels, params, chemical_types)

    formats = {_classify_path(p) for p in flat_paths}
    if len(formats) > 1:
        raise ValueError('Mixing DP and extxyz paths is not supported: %s' % (paths,))

    if formats == {'extxyz'}:
        return ExtXYZDataset(flat_paths, labels, params, chemical_types)
    leaves = [DPDataset(p, labels, params, chemical_types) for p in flat_paths]
    return DatasetGroup(leaves, chemical_types)


class DatasetLeaf:
    """
    Internal dataset class for in-memory data of a single composition.

    Data is stored in the input atom order. The model receives ``type_idx`` and
    handles type sorting internally.
    """
    def __init__(self, labels, params, type_arr, data, paths=None):
        self.chemical_types = getattr(self, 'chemical_types', None)
        type_arr = np.array(type_arr, dtype=int)
        
        # --- NUEVA LÓGICA: Soporte para Buckets 2D (ExtXYZ y Matbench) ---
        if type_arr.ndim == 2:
            self.type_idx = type_arr  # (N_frames, N_atoms)
            self.natoms = type_arr.shape[1]
            self.data = data
            self.nframes = len(self.data['coord'])
            
            # Recuperar matrices de conteo real para el Ebias
            self._frame_type_counts = data.pop('_type_counts')
            self.ntypes = self._frame_type_counts.shape[1]
            self.type_count = self._frame_type_counts.sum(axis=0)
            self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
            
            for l in labels:
                if l == 'energy':
                    self.data[l] = self.data[l].reshape(-1)
            
            self.data['box'] = self.data['box'].reshape(-1, 3, 3)
            self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
            
            self.pointer = self.nframes
            self.nsel = params.get('atomic_sel', None)
            if self.nsel is not None:
                self.nsel = [0]
            if any(['atomic' in l for l in labels]):
                self.nlabels = sum(self.type_count[self.nsel])
            else:
                self.nlabels = self.natoms
                
            if paths is not None:
                from os.path import abspath
                print('# Dataset Leaf (Bucket %d): %d frames. Path:' % (self.natoms, self.nframes),
                      ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))

        # --- LÓGICA ANTERIOR: Para formato DP clásico (1D) ---
        else:
            raw_natoms = len(type_arr)
            real_type_arr = type_arr
            
            self.raw_natoms = raw_natoms
            self.raw_type_arr = real_type_arr
            self.type_count = np.bincount(real_type_arr)
            self.ntypes = len(self.type_count)
            self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
            
            buckets = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024]
            bucket = next((b for b in buckets if b >= raw_natoms), raw_natoms)
            pad_len = bucket - raw_natoms
            
            self.natoms = bucket 
            if pad_len > 0:
                self.type_idx = np.pad(real_type_arr, (0, pad_len), constant_values=-1)
            else:
                self.type_idx = real_type_arr
                
            self.type = self.type_idx
            self.data = data
            self.nframes = len(self.data['coord'])
            
            for l in labels:
                assert self.data[l].shape[0] == self.nframes
            self.pointer = self.nframes
            
            self.nsel = params.get('atomic_sel', None)
            if self.nsel is not None:
                self.nsel = [0]
                
            if any(['atomic' in l for l in labels]):
                self.nlabels = sum(self.type_count[self.nsel])
            else:
                self.nlabels = self.natoms
                
            for l in labels:
                if l in ['coord', 'force']:
                    v = self.data[l].reshape(self.data[l].shape[0], raw_natoms, 3)
                    if pad_len > 0:
                        v = np.pad(v, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
                    self.data[l] = v
                if l == 'energy':
                    self.data[l] = self.data[l].reshape(-1)
                if 'atomic' in l:
                    try:
                        v = self.data[l].reshape(self.data[l].shape[0], raw_natoms, -1)
                        if pad_len > 0:
                            v = np.pad(v, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
                        self.data[l] = v
                    except:
                        pass
                    
            self.data['box'] = self.data['box'].reshape(-1, 3, 3)
            self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
            
            if paths is not None:
                from os.path import abspath
                print('# Dataset loaded: %d frames/%d atoms (Padded from %d raw atoms). Path:' % (self.nframes, self.natoms, raw_natoms),
                      ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))
    def count_max(self):
        return np.array(self.type_count)

    def fill_type(self, ntypes):
        self.type_count = np.pad(self.type_count, (0, ntypes - self.ntypes))

    def _get_stats(self, rcut, bs):
        if not hasattr(self, 'lattice_args'):
            raise AttributeError("lattice_args not set. Call compute_lattice_candidate(rcut) before get_stats.")
        batch = self.get_batch(bs)[0]
        coord, box = batch['coord'], batch['box']
        
        if self.type_idx.ndim == 2:
            # Para buckets 2D extraemos el primer frame sin átomos fantasma para las estadísticas
            real_mask = batch['type_idx'][0] != -1
            coord = coord[0:1, real_mask]
            box = box[0:1]
            real_type_arr = batch['type_idx'][0][real_mask]
            type_count = np.bincount(real_type_arr, minlength=self.ntypes)
            coord = coord[:, np.argsort(real_type_arr, kind='stable')]
        else:
            coord = coord[:, :self.raw_natoms]
            coord = coord[:, np.argsort(self.raw_type_arr, kind='stable')]
            type_count = self.type_count
            
        r_Bnm = vmap(get_relative_coord, (0, 0, None, None))(coord, box, type_count, self.lattice_args)[1]
        sr_BnM = [sr(jnp.concatenate(r, axis=-1), rcut) for r in r_Bnm]
        sr_sum = np.array([s.sum() for s in sr_BnM])
        sr_sum2 = np.array([(s**2).sum() for s in sr_BnM])
        sr_count = np.array([(s > 1e-15).sum() for s in sr_BnM])
        Nnbrs = (np.concatenate(sr_BnM, axis=1) > 0).sum(2).mean() + 1
        return np.array([sr_sum, sr_sum2, sr_count, Nnbrs * np.ones_like(sr_sum)])

    def get_batch(self, batch_size, type='frame'):
        if type == 'label':
            batch_size = int(batch_size / self.nlabels + 1)
        
        # =================================================================
        # MODIFICATION: Dynamic Batch Size (VRAM Protection)
        # Fixes the volume of atoms per batch rather than the number of frames.
        # =================================================================
        vram_atom_budget = 2048  # Límite seguro
        dynamic_bs = max(1, vram_atom_budget // self.natoms)
        
        # Usamos el menor entre el batch solicitado y el dinámico permitido
        actual_bs = min(batch_size, dynamic_bs)
        
        indices = np.arange(self.pointer, self.pointer + actual_bs) % self.nframes
        self.pointer = (self.pointer + actual_bs) % self.nframes
            
        batch = {
            'atomic' if 'atomic' in l else l:
            self.data[l][indices]
            for l in self.data
        }

        # --- Inyectar type_idx limpio según si es 1D o 2D ---
        if self.type_idx.ndim == 2:
            batch['type_idx'] = self.type_idx[indices]
        else:
            batch['type_idx'] = np.tile(self.type_idx, (actual_bs, 1))

        return batch, tuple(self.type_idx.flatten()), self.lattice_args

    def compute_lattice_candidate(self, rcut):
        self.lattice_args = compute_lattice_candidate(self.data['box'], rcut)

    def get_stats(self, rcut, bs):
        self.params = {'rcut': rcut}
        sr_sum, sr_sum2, sr_count, Nnbrs = self._get_stats(rcut, bs)
        sr_sum, sr_sum2, sr_count = sr_sum[self.valid_types], sr_sum2[self.valid_types], sr_count[self.valid_types]
        
        # --- LÓGICA CORREGIDA: Protección contra división por cero ---
        sr_count_safe = np.maximum(sr_count, 1)
        
        self.params['valid_types'] = self.valid_types
        self.params['sr_mean'] = sr_sum / sr_count_safe
        
        # Uso de np.maximum para evitar raíces de números negativos por precisión flotante
        variance = sr_sum2 / sr_count_safe - self.params['sr_mean']**2
        self.params['sr_std'] = np.sqrt(np.maximum(variance, 0.0))
        
        self.params['Nnbrs'] = Nnbrs[0]
        
        # Forzar el conteo químico real para los embeddings de la red
        if self.chemical_types is not None:
            self.params['ntypes'] = len(self.chemical_types)
            self.params['chemical_types'] = self.chemical_types
        else:
            self.params['ntypes'] = self.type_idx.max() + 1
            
        return self.params

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
        type_count = type_count[:, self.valid_types]
        
        # --- LÓGICA CORREGIDA: Resolver Ebias para cada elemento ---
        ebias_valid = np.linalg.lstsq(type_count, energy_mean, rcond=1e-3)[0].astype(np.float32)
        
        real_ntypes = len(self.chemical_types) if self.chemical_types is not None else self.ntypes
        ebias_full = np.zeros(real_ntypes, dtype=np.float32)
        ebias_full[self.valid_types] = ebias_valid
        
        return ebias_full

    def get_atomic_label_scale(self):
        label = [label for label in self.data.keys() if 'atomic' in label][0]
        return np.std(self.data[label])

    def get_flattened_data(self):
        return [{'data': self.data, 'type_idx': self.type_idx, 'lattice_args': self.lattice_args}]

    def get_leaves(self):
        return [self]


class DPDataset(DatasetLeaf):
    """
    Dataset for DeepMD (DP) format directories.

    Loads data from DP training directories containing type.raw and set.*/ subdirs
    with .npy files. Concatenates data across all sets in the directory.
    """
    def __init__(self, path, labels, params=None, chemical_types=None):
        self.chemical_types = tuple(chemical_types) if chemical_types else None
        type_arr = np.genfromtxt(path + '/type.raw').astype(int)
        data = {
            l: np.concatenate([np.load(s + l + '.npy') for s in sorted(glob(path + '/set.*/'))])
            for l in labels
        }
        super().__init__(labels, params or {}, type_arr, data, paths=[path])


class DatasetGroup:
    """
    Composite dataset made from multiple subset datasets.

    A DatasetGroup represents a mixture of DatasetLeaf subsets. Sampling across
    subsets is weighted by subset size, stored in self.prob, so larger subsets are
    selected more often during batch generation.
    """
    def __init__(self, subsets, chemical_types=None):
        self.subsets = subsets
        self.chemical_types = tuple(chemical_types) if chemical_types else None
        self.nframes = sum([subset.nframes for subset in self.subsets])
        self.ntypes = max([subset.ntypes for subset in self.subsets])
        [subset.fill_type(self.ntypes) for subset in self.subsets]
        self.prob = np.array([subset.nframes for subset in self.subsets]) / self.nframes
        self.type_count = self.count_max()
        self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
        if self.chemical_types is None:
            cts = {s.chemical_types for s in self.subsets if s.chemical_types is not None}
            if len(cts) > 1:
                raise ValueError('Inconsistent chemical_types across subsets: %s' % cts)
            if cts:
                self.chemical_types = cts.pop()

    def count_max(self):
        return np.array([subset.count_max() for subset in self.subsets]).max(0)

    def fill_type(self, ntypes):
        for subset in self.subsets:
            subset.fill_type(ntypes)

    def _get_stats(self, rcut, bs):
        s = np.stack([subset._get_stats(rcut, bs) for subset in self.subsets], axis=-1)
        return (s * self.prob).sum(-1)

    def get_batch(self, batch_size, type='frame'):
        subset = np.random.choice(len(self.subsets), p=self.prob)
        return self.subsets[subset].get_batch(batch_size, type)

    def compute_lattice_candidate(self, rcut):
        for subset in self.subsets:
            subset.compute_lattice_candidate(rcut)
            
        # =================================================================
        # MODIFICATION: Unify lattice_args across all subsets
        # This prevents recompilation when switching between different leaves.
        # =================================================================
        best_subset = max(self.subsets, key=lambda s: s.lattice_args['lattice_max'])
        global_lattice_args = best_subset.lattice_args
        for subset in self.subsets:
            subset.lattice_args = global_lattice_args

    def get_stats(self, rcut, bs):
        self.params = {'rcut': rcut}
        sr_sum, sr_sum2, sr_count, Nnbrs = self._get_stats(rcut, bs)
        sr_sum, sr_sum2, sr_count = sr_sum[self.valid_types], sr_sum2[self.valid_types], sr_count[self.valid_types]
        
        # --- LÓGICA CORREGIDA: Protección contra división por cero ---
        sr_count_safe = np.maximum(sr_count, 1)
        
        self.params['valid_types'] = self.valid_types
        self.params['sr_mean'] = sr_sum / sr_count_safe
        
        # Uso de np.maximum para evitar raíces de números negativos por precisión flotante
        variance = sr_sum2 / sr_count_safe - self.params['sr_mean']**2
        self.params['sr_std'] = np.sqrt(np.maximum(variance, 0.0))
        
        self.params['Nnbrs'] = Nnbrs[0]
        
        # Forzar el conteo químico real para los embeddings de la red
        if self.chemical_types is not None:
            self.params['ntypes'] = len(self.chemical_types)
            self.params['chemical_types'] = self.chemical_types
        else:
            self.params['ntypes'] = self.type_idx.max() + 1
            
        return self.params

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
        type_count = type_count[:, self.valid_types]
        
        # --- LÓGICA CORREGIDA: Resolver Ebias para cada elemento ---
        ebias_valid = np.linalg.lstsq(type_count, energy_mean, rcond=1e-3)[0].astype(np.float32)
        
        real_ntypes = len(self.chemical_types) if self.chemical_types is not None else self.ntypes
        ebias_full = np.zeros(real_ntypes, dtype=np.float32)
        ebias_full[self.valid_types] = ebias_valid
        
        return ebias_full

    def get_atomic_label_scale(self):
        return (np.array([subset.get_atomic_label_scale() for subset in self.subsets]) * np.array(self.prob)).sum()

    def _get_energy_stats(self):
        return sum([subset._get_energy_stats() for subset in self.subsets], [])

    def get_flattened_data(self):
        return sum([subset.get_flattened_data() for subset in self.subsets], [])

    def get_leaves(self):
        return sum([s.get_leaves() for s in self.subsets], [])


class ExtXYZDataset(DatasetGroup):
    """
    Dataset for extended XYZ (.xyz/.extxyz) files.
    Optimized for extremely diverse datasets (Matbench/MPTraj) using Bucket Hashing.
    """
    def __init__(self, paths, labels, params=None, chemical_types=None):
        raw_frames = []
        all_zs = set()
        for path in paths:
            atoms_list = read(path, index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            for atoms in atoms_list:
                zs = np.asarray(atoms.get_atomic_numbers(), dtype=int)
                all_zs.update(zs.tolist())
                entry = {'_zs': zs}
                for l in labels:
                    if l == 'box':
                        entry['box'] = np.asarray(atoms.get_cell().array, dtype=np.float32)
                    elif l == 'coord':
                        entry['coord'] = np.asarray(atoms.get_positions(), dtype=np.float32)
                    elif l == 'force':
                        entry['force'] = np.asarray(atoms.get_forces(), dtype=np.float32)
                    elif l == 'energy':
                        entry['energy'] = np.asarray(atoms.get_potential_energy(), dtype=np.float32)
                    else:
                        if l in atoms.arrays:
                            entry[l] = np.asarray(atoms.arrays[l], dtype=np.float32)
                        elif l in atoms.info:
                            entry[l] = np.asarray(atoms.info[l], dtype=np.float32)
                        else:
                            raise ValueError('Label %s not found in extxyz frame from %s' % (l, path))
                raw_frames.append(entry)

        if chemical_types is None:
            chemical_types = tuple(sorted(all_zs))
        else:
            unknown = all_zs - set(chemical_types)
            if unknown:
                print(f"# ⚛️ New elements discovered! Expanding table: {sorted(unknown)}")
                chemical_types = tuple(list(chemical_types) + sorted(unknown))
                
        self.chemical_types = chemical_types
        z_to_idx = {z: i for i, z in enumerate(chemical_types)}

        # =================================================================
        # MODIFICATION: O(1) Bucket Hashing for Matbench/MPTraj
        # Group by padded bucket size instead of exact chemical sequence.
        # =================================================================
        groups = {}
        buckets = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024]

        for entry in raw_frames:
            zs = entry.pop('_zs')
            types = np.array([z_to_idx[int(z)] for z in zs], dtype=int)
            raw_natoms = len(types)
            
            bucket = next((b for b in buckets if b >= raw_natoms), raw_natoms)
            pad_len = bucket - raw_natoms
            
            bucket = next((b for b in buckets if b >= raw_natoms), raw_natoms)
            pad_len = bucket - raw_natoms
            
            # Guardar el conteo de elementos reales de ESTE frame para el Ebias
            type_count = np.bincount(types, minlength=len(chemical_types))
            entry['_type_count'] = type_count
            
            # Aplicar padding individualmente para que NumPy deje agruparlos
            if pad_len > 0:
                types = np.pad(types, (0, pad_len), constant_values=-1)
                for l in labels:
                    if l in ['coord', 'force']:
                        entry[l] = np.pad(entry[l], ((0, pad_len), (0, 0)), mode='constant')
                    elif 'atomic' in l:
                        entry[l] = np.pad(entry[l], ((0, pad_len), (0, 0)), mode='constant')
            
            grp = groups.setdefault(bucket, {'type': [], 'frames': []})
            grp['type'].append(types)
            grp['frames'].append(entry)

        subsets = []
        for bucket, grp in groups.items():
            frames = grp['frames']
            data = {l: np.stack([f[l] for f in frames]) for l in labels}
            data['_type_counts'] = np.stack([f['_type_count'] for f in frames])
            
            type_arr_2d = np.stack(grp['type'])
            subsets.append(DatasetLeaf(labels, params or {}, type_arr_2d, data))

        super().__init__(subsets, chemical_types=chemical_types)
        print('# Dataset loaded (extxyz bucketed): %d frames grouped into %d bucket(s). Path:'
              % (len(raw_frames), len(subsets)),
              ''.join(['\n# \t\'%s\'' % abspath(p) for p in paths]))
    


def compute_lattice_candidate(boxes, rcut, print_info=True, disable_ortho=False):
    N = 2  # This algorithm is heuristic and subject to change. Increase N in case of missing neighbors.
    ortho = not vmap(lambda box: box - jnp.diag(jnp.diag(box)))(boxes).any()
    recp_norm = jnp.linalg.norm((jnp.linalg.inv(boxes)), axis=-1)
    n = np.ceil(rcut * recp_norm - 0.5).astype(int).max(0)
    lattice_cand = jnp.stack(
        np.meshgrid(range(-n[0], n[0] + 1), range(-n[1], n[1] + 1), range(-n[2], n[2] + 1), indexing='ij'),
        axis=-1).reshape(-1, 3)
    trial_points = jnp.stack(np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), np.arange(-N, N + 1)),
                             axis=-1).reshape(-1, 3) / (2 * N)
    is_neighbor = jnp.linalg.norm((lattice_cand[:, None] - trial_points)[None] @ boxes[:, None], axis=-1) < rcut
    lattice_cand = np.array(lattice_cand[is_neighbor.any((0, 2))])
    lattice_max = is_neighbor.sum(1).max().item()
    if print_info:
        print('# Lattice vectors for neighbor images: Max %d out of %d candidates.' % (lattice_max, len(lattice_cand)))
    return {'lattice_cand': tuple(map(tuple, lattice_cand)),
            'lattice_max': lattice_max,
            'ortho': ortho if not disable_ortho else False}
