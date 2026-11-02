"""
utils Datasets
General data reading datasets from csv files into different forms.
"""
import pandas as pd
import torch
import os

from OAT.pose_extraction import (
    canonicalize,
    estimate_heading,
    circular_offset,
    fit_body_template,
    procrustes_se2,
    wrap
)

DEFAULT_PART_NAMES = [
    "base",             # 0   non-physical pathfinding joint
    "spine_base",       # 1
    "spine_mid",        # 2
    "spine_front",      # 3
    "neck",             # 4
    "head",             # 5
    "nose",             # 6
    "ear_l",            # 7
    "ear_r",            # 8
    "shoulder_l",       # 9
    "elbow_l",          # 10
    "wrist_l",          # 11
    "shoulder_r",       # 12
    "elbow_r",          # 13
    "wrist_r",          # 14
    "hip_l",            # 15
    "knee_l",           # 16
    "ankle_l",          # 17
    "hindpaw_l",        # 18
    "hip_r",            # 19
    "knee_r",           # 20
    "ankle_r",          # 21
    "hindpaw_r",        # 22
    "tail_base",        # 23
    "tail_mid",         # 24
    "tail_end",         # 25
]

# Indices below are in BASE-EXCLUDED space, i.e. DEFAULT_PART_NAMES[i+1].
# Written as (parent, child). Root is spine_base (0).
DEFAULT_RIGID_EDGES = [
    [0, 1],    # spine_base   -> spine_mid
    [1, 2],    # spine_mid    -> spine_front
    [2, 3],    # spine_front  -> neck
    [3, 4],    # neck         -> head
    [4, 5],    # head         -> nose
    [4, 6],    # head         -> ear_l
    [4, 7],    # head         -> ear_r
    [2, 8],    # spine_front  -> shoulder_l
    [8, 9],    # shoulder_l   -> elbow_l
    [9, 10],   # elbow_l      -> wrist_l
    [2, 11],   # spine_front  -> shoulder_r
    [11, 12],  # shoulder_r   -> elbow_r
    [12, 13],  # elbow_r      -> wrist_r
    [0, 14],   # spine_base   -> hip_l
    [14, 15],  # hip_l        -> knee_l
    [15, 16],  # knee_l       -> ankle_l
    [16, 17],  # ankle_l      -> hindpaw_l
    [0, 18],   # spine_base   -> hip_r
    [18, 19],  # hip_r        -> knee_r
    [19, 20],  # knee_r       -> ankle_r
    [20, 21],  # ankle_r      -> hindpaw_r
    [0, 22],   # spine_base   -> tail_base
    [22, 23],  # tail_base    -> tail_mid
    [23, 24],  # tail_mid     -> tail_end
]

# base-excluded indices, for readability
J = {name: i - 1 for i, name in enumerate(DEFAULT_PART_NAMES) if i > 0}
AXIAL_INDICES = [J["spine_base"], J["spine_mid"], J["spine_front"], J["neck"], J["head"]]
PAW_INDICES = [J["wrist_l"], J["wrist_r"], J["hindpaw_l"], J["hindpaw_r"]]

def build_parent_list(edges, N):
    """(E,2) edges -> parent list of length N, root = -1. Root is inferred."""
    parent = [-1] * N
    children = set()
    for p, c in edges:
        parent[int(c)] = int(p)
        children.add(int(c))
    roots = [i for i in range(N) if i not in children]
    assert len(roots) == 1, f"skeleton must have exactly one root, found {roots}"
    # sanity: every non-root reachable from root
    seen, stack = set(), [roots[0]]
    adj = {i: [] for i in range(N)}
    for p, c in edges:
        adj[int(p)].append(int(c))
        adj[int(c)].append(int(p))
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u])
    assert len(seen) == N, f"skeleton is disconnected: {sorted(set(range(N)) - seen)} unreachable"
    return parent

class RawDataset:
    """
    Loads the CSV and produces, in one consistent frame:

        base            (T, 3)     world path of the base joint
        deform_global   (T, N, 3)  world deformable
        rigid_global    (T, N, 3)  world rigid
        heading_gt      (T,)       true path yaw (from the rigid skeleton)
        deform_local    (T, N, 3)  canonicalized deformable  <- MAIN TARGET
        rigid_local     (T, N, 3)  canonicalized rigid
    """

    def __init__(self, config=None):
        # Configs
        self.config = config or {}
        self.file_path = self.config['file_path']
        self.frames_file = self.config.get("frames_file", None)
        self.part_names = self.config.get("part_names", DEFAULT_PART_NAMES)
        self.fps = self.config.get("fps", 30)
        self.load_template = config.get("template_load_path", None)
        self.save_template = config.get("template_save_path", None)
        self.edges = config.get("edges", DEFAULT_RIGID_EDGES)

        # Load data
        raw = pd.read_csv(self.file_path)
        raw = raw.drop(columns=[c for c in ("frame", "joint_id") if c in raw.columns])
        present = list(raw["part"].unique())
        self.N = len(present) - 1
        n_slots = self.N + 1
        assert len(raw) % n_slots == 0, (
            f"row count {len(raw)} is not divisible by joints-per-frame {n_slots}; "
            "the CSV has ragged frames"
        )

        # Reorder parts
        part_order = {p: i for i, p in enumerate(self.part_names)}
        missing = set(present) - set(part_order)
        assert not missing, f"parts in CSV not present in part_names: {missing}"
        raw["_ord"] = raw["part"].map(part_order)
        raw = raw.sort_values(["time", "_ord"], kind="mergesort").drop(columns=["_ord"])
        self.part_names = [p for p in self.part_names if p in present]


        def col3(a, b, c):
            return torch.tensor(raw[[a, b, c]].to_numpy(), dtype=torch.float32).reshape(-1, n_slots, 3)

        deform_raw = col3("x_d", "y_d", "z_d")
        self.base = deform_raw[:, 0]                    # (T, 3)
        self.deform_global = deform_raw[:, 1:]          # (T, N, 3)
        self.rigid_global = col3("x_r", "y_r", "z_r")[:, 1:]
        self.deform_local = col3("x_l", "y_l", "z_l")[:, 1:]
        self.T = self.deform_global.shape[0]
        self.time = raw["time"].to_numpy()
        self.parts = raw["part"].to_numpy()
        
        # Heading
        self.heading = estimate_heading(self.rigid_global, lr_indices=(J["hip_l"], J["hip_r"]), nose_idx=J["nose"], tail_idx=J["tail_base"])
        self.rigid_local = canonicalize(self.rigid_global, self.heading, self.base)

        # Behavioural data
        self.loco_clip = None
        self.ovl_clip = None
        if self.frames_file is not None:
            info = pd.read_csv(self.frames_file)
            if "loco_clip" in info:
                self.loco_clip = info["loco_clip"].to_numpy()
                assert len(self.loco_clip) == self.T, f"frames_file has {len(self.loco_clip)} rows, CSV has {self.T} frames"
            if "ovl_clip" in info:
                self.ovl_clip = info["ovl_clip"].to_numpy()
                assert len(self.ovl_clip) == self.T, f"frames_file has {len(self.ovl_clip)} rows, CSV has {self.T} frames"

        # Connectivity
        self.edges = torch.tensor(self.edges, dtype=torch.long)
        self.parent = build_parent_list(self.edges, self.N)
        self.adj = torch.zeros(self.N, self.N, dtype=torch.uint8)
        self.adj[self.edges[:, 0], self.edges[:, 1]] = 1
        self.adj[self.edges[:, 1], self.edges[:, 0]] = 1

        #  SE(2) body-frame estimator
        if self.load_template:
            self.template_xy = torch.load(self.load_template + "/template.pt")
            self.joint_weights = torch.load(self.load_template + "/weights.pt")
        else:
            self._fit_body_frame(config.get("frame_estimator", {}))
        if self.save_template:
            os.makedirs(self.save_template, exist_ok=True)
            torch.save(self.template_xy, self.save_template + "/template.pt")
            torch.save(self.joint_weights, self.save_template + "/weights.pt")
        

    def _joint_index(self, name):
        """Index in BASE-EXCLUDED space, resolved against this CSV's parts."""
        names = list(self.part_names)
        if names and names[0] == "base":
            names = names[1:]
        if name in names:
            return names.index(name)
        raise KeyError(
            f"joint '{name}' not in this recording's parts {names}. Set "
            "config['frame_estimator']['axis_pair'] to a pair that exists."
        )

    def _fit_body_frame(self, cfg):
        """
        Fit the canonical template and the per-joint reliability weights.
        The estimator itself is a weighted 2D Procrustes onto this template
        """
        ax = cfg.get("axis_pair", ("tail_base", "nose"))
        axis_pair = tuple(self._joint_index(a) if isinstance(a, str) else int(a) for a in ax)
        verbose = bool(cfg.get("verbose", True))

        fit = fit_body_template(
            self.deform_global[..., :2],
            init_theta=self.heading,
            axis_pair=axis_pair,
            n_iter=int(cfg.get("n_iter", 6)),
            weight_clip=float(cfg.get("weight_clip", 50.0)),
            verbose=verbose,
            tag=" deform",
        )
        self.template_xy = fit["template"]          # (N, 2)
        self.joint_weights = fit["weights"]         # (N,)
        self.frame_theta = fit["theta"]             # (T,)   full-visibility fit
        self.frame_xy = fit["t"]                    # (T, 2)
        self.frame_fit_report = fit["report"]
        self.axis_pair = axis_pair

        # Ground truth: the rigid keypoints carry ONLY the global transform, so a Procrustes fit on them recovers the true body frame essentially
        rigid_fit = fit_body_template(
            self.rigid_global[..., :2],
            init_theta=self.heading,
            axis_pair=axis_pair,
            n_iter=int(cfg.get("n_iter", 6)),
            verbose=False,
        )
        self.frame_theta_rigid = rigid_fit["theta"]
        self.frame_xy_rigid = rigid_fit["t"]

        # if verbose:
        #     r = self.frame_error_report()
        #     print(
        #         f"[Dataset] body frame vs rigid GT: "
        #         f"yaw RMS {r['yaw_rms_deg']:.3f} deg, "
        #         f"xy RMS {r['xy_rms']:.4f}"
        #     )