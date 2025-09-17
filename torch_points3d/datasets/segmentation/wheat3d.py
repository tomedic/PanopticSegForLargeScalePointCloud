import os
import os.path as osp
import numpy as np
import torch
import random
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data
import logging
from sklearn.neighbors import KDTree

# PLY reader
from torch_points3d.modules.KPConv.plyutils import read_ply
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

WHEAT3D_NUM_CLASSES = 2

INV_OBJECT_LABEL = {
    0: "background",
    1: "wheathead",
}

OBJECT_COLOR = np.asarray(
    [
        [95, 156, 196],  # 'ground' .-> . blue
        [233, 229, 107],  # 'wheathead' .-> .yellow
        # [0, 0, 0],  # unlabelled .->. black

    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

################################### UTILS #######################################
def object_name_to_label(object_class):
    """if labels within .ply are strings corresponding to INV_OBJECT_LABEL values/names,
     convert from name in INV_OBJECT_LABEL to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label


def read_raw_wheat3d_format(train_file, label_out=True, verbose=False, debug=False):
    """extract data from data/wheat3d/raw/*.ply file, return as tuple of torch.Tensors xyz,
     semantic_labels, instance_labels"""

    # check if the file is a .ply file
    if not train_file.endswith(".ply"):
        raise ValueError(f"File {train_file} is not a .ply file")

    # check if the file exists
    if not os.path.exists(train_file):
        raise ValueError(f"File {train_file} does not exist")

    raw_path = train_file
    data = read_ply(raw_path)
    xyz = np.vstack((data["x"], data["y"], data["z"])).astype(np.float32).T
    if not label_out:
        return xyz
    semantic_labels = data["scalar_classes"].astype(np.int64) - 1
    instance_labels = data["scalar_instances"].astype(np.int64) + 1

    # TODO: add optional extraction of extra features (Intensity, RGB, etc.)

    return (
        torch.from_numpy(xyz),
        torch.from_numpy(semantic_labels),
        torch.from_numpy(instance_labels),
    )


def to_ply(pos, label, file):
    """save wheat3d semantic class predictions to disk using wheat3d color scheme
     defined in OBJECT_COLOR"""
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "Wheat3D")
    PlyData([el], byte_order=">").write(file)


def to_eval_ply(pos, pre_label, gt, file):
    """save wheat3d semantic class predictions and ground truth to disk using wheat3d color scheme
     defined in OBJECT_COLOR"""
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "u16"), ("gt", "u16")])
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    PlyData.write(file)


def to_ins_ply(pos, label, file):
    """save wheat3d instance predictions to disk using random colors"""
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32) + 1
    rd_colors = np.random.randint(255, size=(max_instance, 3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    PlyData.write(file)


################################### Used for fused NPM3D radius sphere ###################################


class Wheat3d(InMemoryDataset):
    """Wheat3d in memory dataset. Takes all .ply files in the specified data/dataset_name/raw/ folder
    (hydra .yaml config) and fuses them into a single Wheat3d(InMemoryDataset) object. Scene parts are
     to be split into train, val, test part of the scene using _train, _val, _test suffixes. Ply files are
     expected to have scalar fields for semantic labels and instance labels ("classes" and "instances").
     Having other (optional) fields is ok, but using them is currently not supported (e.g. intensity, RGB, etc.)

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int or None
        number between 1 and 4 that denotes the area (scene) used for testing, or None if using all scenes
        for testing and training (according to split parameter - name suffixes _train, _val, _test)
    split: str
        can be one of train, trainval, val or test (automatically inferred)
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    num_classes = WHEAT3D_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        test_area=None,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        # Accept test_area from configs but do not rely on predefined areas
        # - None: ignore areas entirely (suffix-only splitting)
        # - int: kept for compatibility, but unused in processing
        # - list[str]: explicit file list for eval (handled by process_test)
        if test_area is None:
            self.area_name = None
        elif not isinstance(test_area, int):
            # When provided a list of file paths (eval on custom fold)
            assert len(test_area) >= 1
            self.area_name = os.path.split(test_area[0])[-1].split(".")[0]
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size

        # inherit basic functionality from InMemoryDataset
        super(Wheat3d, self).__init__(root, transform, pre_transform, pre_filter)
        # define paths to processed data according to split
        if isinstance(test_area, int) or test_area is None:
            if split == "train":
                path = self.processed_paths[0]
            elif split == "val":
                path = self.processed_paths[1]
            elif split == "test":
                path = self.processed_paths[2]
            elif split == "trainval":
                path = self.processed_paths[3]
            else:
                raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
            # load data from path
            self._load_data(path)
            # if split is test, load all data from raw_areas_paths[0] -> single fused raw file in .pt format (for eval.py use)
            if split == "test":
                # Single fused raw file
                self.raw_test_data = torch.load(self.raw_areas_paths[0])
        else:
            # self.process_test(test_area)
            path = self.processed_paths[0]
            self._load_data(path)
            self.raw_test_data = torch.load(self.raw_areas_paths[0])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        # Dynamically list all .ply files found in data/dataset_name/raw/
        if not osp.isdir(self.raw_dir):
            return []
        return sorted(
            [
                osp.join(self.raw_dir, f)
                for f in os.listdir(self.raw_dir)
                if f.lower().endswith(".ply")
            ]
        )

    @property
    def processed_dir(self):
        # Creates/defines a processed directory according to the grid size (currently only hyperparameter that is traced)
        return osp.join(self.root, "processed_" + str(self.grid_size))

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        # Single fused raw file containing all data from data/dataset_name/raw/ (not only predefined areas)
        return [os.path.join(self.processed_dir, "raw_fused.pt")]

    @property
    def processed_file_names(self):
        # Generic filenames according to split / file suffixes (independent of area/test_area)
        return [
            "train.pt",
            "val.pt",
            "test.pt",
            "trainval.pt",
            *[osp.basename(p) for p in self.raw_areas_paths],
            osp.basename(self.pre_processed_path),
        ]

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0

    # def download(self):
    #    super().download()

    def process(self):
        # Process raw .ply files, if not already processed
        if not os.path.exists(self.pre_processed_path):
            # Gather all .ply files under raw_dir
            input_ply_files = self.raw_file_names

            # Single fused group containing all data from data/dataset_name/raw/ (not only predefined areas)
            data_list = [[]]

            for file_path in input_ply_files:
                fname = os.path.basename(file_path)
                # Keep only files that match expected split suffixes
                is_val = fname.endswith("_val.ply")
                is_test = fname.endswith("_test.ply")
                is_train = fname.endswith("_train.ply")
                is_testtrain = fname.endswith("_testtrain.ply")
                if not (is_val or is_test or is_train or is_testtrain):
                    continue

                # read .ply file
                xyz, semantic_labels, instance_labels = read_raw_wheat3d_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )
                # create Data object (PyG - PyTorch Geometric)
                data = Data(pos=xyz, y=semantic_labels)
                # Flags used later to split into datasets (validation, test, train, testtrain)
                data.validation_set = is_val
                data.test_set = is_test or is_testtrain
                # keep instance labels (for instance segmentation) if desired
                if self.keep_instance:
                    data.instance_labels = instance_labels

                # Check if data meets pre_filter criteria (PyG - PyTorch Geometric) if provided, if not, skip this data
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                data_list[0].append(data)

            # fuse data_list into a single Data object with raw/unprocessed point clouds (PyG - PyTorch Geometric),
            # TODO:if different areas (scenes) defined, should be split into different Data objects (currently not fully implemented)
            # Fuse all raw inputs into a single Data for downstream use
            if len(data_list) == 1:
                raw_fused = cT.PointCloudFusion()(data_list[0])
            else:
                raw_fused_list = cT.PointCloudFusion()(data_list)
                # In current wheat3d usage we expect a single fused space
                # If multiple, concatenate them into one Batch and drop batch attrs
                if isinstance(raw_fused_list, list):
                    from torch_geometric.data import Batch

                    fused = Batch.from_data_list(raw_fused_list)
                    if hasattr(fused, "batch"):
                        delattr(fused, "batch")
                    if hasattr(fused, "ptr"):
                        delattr(fused, "ptr")
                    raw_fused = fused
                else:
                    raw_fused = raw_fused_list
            torch.save(raw_fused, self.raw_areas_paths[0])

            # Apply pre_transform per fused group (all data or per-scene data) - if pre_transform provided
            for area_datas in data_list:
                if self.pre_transform is not None:
                    area_datas = self.pre_transform(area_datas)
            # save pre_processed data to .pt file
            torch.save(data_list, self.pre_processed_path)
        else:
            # load pre_processed data from .pt file, if raw data was already processed and saved to .pt file
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        # split data into train, val, trainval, test datasets
        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        # Implemented for single fused group at index 0 (all data)
        # TODO: if different areas (scenes) defined, should be split into different Data objects (currently not fully implemented)
        for data in data_list[0]:
            validation_set = getattr(data, "validation_set", False)
            if hasattr(data, "validation_set"):
                del data.validation_set
            test_set = getattr(data, "test_set", False)
            if hasattr(data, "test_set"):
                del data.test_set
            if validation_set:
                val_data_list.append(data)
            elif test_set:
                test_data_list.append(data)
            else:
                train_data_list.append(data)
        trainval_data_list = val_data_list + train_data_list

        # print data lists (for debugging)
        print("train_data_list:")
        print(train_data_list)
        print("test_data_list:")
        print(test_data_list)
        print("val_data_list:")
        print(val_data_list)
        print("trainval_data_list:")
        print(trainval_data_list)

        # Apply pre_collate_transform if provided
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        # save data to .pt files
        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def process_test(self, test_area):

        test_data_list = []

        for i, file_path in enumerate(test_area):
            area_name = os.path.split(file_path)[-1]
            if not os.path.exists(self.pre_processed_path):
                xyz, semantic_labels, instance_labels = read_raw_wheat3d_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )
                data = Data(pos=xyz, y=semantic_labels)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_name:")
                print(area_name)
                print("data:")
                print(data)
                test_data_list.append(data)
                # if self.pre_transform is not None:
                #     for data in test_data_list:
                #         data = self.pre_transform(data)
                # torch.save(data, pre_processed_path)
                torch.save(data, self.pre_processed_path)

            else:
                data = torch.load(self.pre_processed_path)
                test_data_list.append(data)

        raw_areas = cT.PointCloudFusion()(test_data_list)
        torch.save(raw_areas, self.raw_areas_paths[0])

        if self.debug:
            return

        print("test_data_list:")
        print(test_data_list)

        # Apply pre_collate_transform if provided to test data
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)

        torch.save(test_data_list, self.processed_paths[0])

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class Wheat3dSphere(Wheat3d):
    """Small variation of Wheat3d that allows random sampling of spheres within an area (scene) during
    training and validation. Spheres radius is specified in the hydra .yaml config. If sample_per_epoch is not specified, spheres
    are not randomly sampled, but taken on a grid with step size equal to sphere radius (used for test time inference).

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int or None
        number between 1 and 4 that denotes the area used for testing, or None if using all scenes
        for testing and training (according to split parameter - name suffixes _train, _val, _test)
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=0.25, grid_size=0.003, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=grid_size, mode="last")
        super().__init__(root, grid_size, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        # For wheat3d: if test_area is None or int, run the standard processing that uses suffixes.
        # Only when an explicit list of files is provided (eval on custom fold) run process_test.
        if self.test_area is None or isinstance(self.test_area, int):
            super().process()
        else:
            super().process_test(self.test_area)

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            # print(self._datas)
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class Wheat3dCylinder(Wheat3dSphere):
    """ Small variation of Wheat3dSphere that allows random sampling of cylinders instead of spheres."""
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
        return cylinder_sampler(area_data)

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class Wheat3dDataset(BaseDataset):
    """Wrapper around Wheat3dSphere/Wheat3dCylinder that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = Wheat3dCylinder if sampling_format == "cylinder" else Wheat3dSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @staticmethod
    def to_ply(pos, label, file):
        """Allows to save Wheat3d predictions to disk using the predefined color scheme (see to_ply function at the top of the file).

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        # from torch_points3d.metrics.s3dis_tracker import S3DISTracker
        # return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
