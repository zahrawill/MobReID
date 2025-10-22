# multicam_proc_fusion.py
"""
Four-camera pipeline with a single global ID assigne* running in a
fusion process. Each camera worker performs:

  1. Frame grab
  2. YOLO detect-only (no external tracker)
  3. FastReID feature extraction per person box
  4. Sends (cam_id, ts, frame, boxes, feats) to the fusion process

The fusion process maintains one `GlobalIDAssigner` that sees detections
from all cameras. It assigns global IDs (gid), updates a per-camera
`tracked` store, draws overlays, and publishes a final summary back
to the parent.

Designed for Linux/Ubuntu ("fork" start method). Toggle show=True if you don't
want live preview windows.
"""

from __future__ import annotations
import matplotlib.cm as cm
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import torch.multiprocessing as tmp
import cv2 as cv

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment  # type: ignore
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D   # registers the “3d” projection
from ultralytics import YOLO
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, set_start_method
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
new_width = 640
new_height = 480
new_dimensions = (new_width, new_height)

# =============================================================================
# Appearance & geometry utilities
# =============================================================================

def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    feat = 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    #nfeat = torch.nn.functional.normalize(feat, dim=0).item()
    return feat


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


@dataclass
class Track:
    gid: int
    bbox: np.ndarray           # last bbox
    feat: torch.Tensor         # last feature
    cam_id: int
    last_ts: float
    age: int = 0
    misses: int = 0


class GlobalIDAssigner:
    """Single assigner across all cameras.

    Cost = alpha * appearance + (1-alpha) * (1 - IoU [same cam only]).
    We only use IoU to help within-camera continuity; cross-camera matches rely
    on appearance.
    """
    #old thres: 0.32
    def __init__(self, dist_th: float = 0.75, alpha: float = 0.95, max_misses: int = 150):
        self.dist_th = float(dist_th)
        self.alpha = float(alpha)
        self.max_misses = int(max_misses)
        self.tracks: List[Track] = []
        self.next_gid: int = 0

    def _build_cost(self, dets: List[Tuple[int, np.ndarray, torch.Tensor]]):
        """Return cost matrix and mapping arrays.

        dets: list of (cam_id, bbox, feat)
        """
        if not self.tracks or not dets:
            return np.empty((0, 0), dtype=np.float32), [], []

        T = len(self.tracks)
        D = len(dets)
        cost = np.zeros((T, D), dtype=np.float32)

        for ti, trk in enumerate(self.tracks):
            for di, (cam_id, bb, fv) in enumerate(dets):
                dist_app = cosine_dist(fv, trk.feat)
                if cam_id == trk.cam_id:
                    dist_iou = 1.0 - iou_xyxy(bb, trk.bbox)
                else:
                    dist_iou = 1.0  # no IoU help across cameras
                cost[ti, di] = self.alpha * dist_app + (1.0 - self.alpha) * dist_iou
        return cost, list(range(T)), list(range(D))

    def update(self, dets: List[Tuple[int, np.ndarray, torch.Tensor]], ts: float) -> List[int]:
        """Assign global IDs to the given detections.

        Returns: list of gids aligned with `dets` order.
        """
        if not dets:
            for trk in self.tracks:
                trk.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
            return []

        cost, _, _ = self._build_cost(dets)
        assigned = [-1] * len(dets)

        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            used_t, used_d = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] > self.dist_th:
                    continue
                cam_id, bb, fv = dets[c]
                trk = self.tracks[r]
                trk.bbox = bb
                trk.feat = fv
                trk.cam_id = cam_id
                trk.last_ts = ts
                trk.age += 1
                trk.misses = 0
                assigned[c] = trk.gid
                used_t.add(r); used_d.add(c)

            # age unmatched
            for idx, trk in enumerate(self.tracks):
                if idx not in used_t:
                    trk.misses += 1
            # remove dead
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

            # create new for remaining dets
            for di, (cam_id, bb, fv) in enumerate(dets):
                if di in used_d:
                    continue
                gid = self.next_gid; self.next_gid += 1
                self.tracks.append(Track(gid, bb, fv, cam_id, ts))
                assigned[di] = gid
        else:
            # no existing tracks: all new
            for di, (cam_id, bb, fv) in enumerate(dets):
                gid = self.next_gid; self.next_gid += 1
                self.tracks.append(Track(gid, bb, fv, cam_id, ts))
                assigned[di] = gid
        return assigned

# =============================================================================
# FastReID wrapper
# =============================================================================
class FastReIDExtractor:
    def __init__(self, cfg_path: str, weight_path: str):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(["MODEL.WEIGHTS", weight_path])
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)
        self.size_wh = tuple(self.predictor.cfg.INPUT.SIZE_TEST[::-1])  # (w, h)

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr.size == 0:
            return torch.zeros(512)  # fallback length; adjust to your model
        rgb = cv.resize(img_bgr[:, :, ::-1], self.size_wh, interpolation=cv.INTER_CUBIC)
        ten = torch.as_tensor(rgb.astype(np.float32).transpose(2, 0, 1))[None]
        feat = self.predictor(ten)
        return feat.squeeze(0).cpu()

# =============================================================================
# Camera worker – detect & extract features, send to fusion
# =============================================================================

def camera_worker(cam_id: int,
                  video_path: str,
                  cfg_path: str,
                  weight_path: str,
                  out_queue: Queue,
                  stop_event: Event) -> None:
    model = YOLO("yolo11n.pt")
    extractor = FastReIDExtractor(cfg_path, weight_path)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Cam {cam_id}] ERROR opening {video_path}")
        out_queue.put(("eof", cam_id))
        return

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    print(f"[Cam {cam_id}] started -> {video_path}")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            frame_idx += 1

            r = model(frame, verbose=False, classes = [0])[0]
            if r.boxes is None or len(r.boxes) == 0:
                out_queue.put(("data", cam_id, ts, frame, [], []))
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()
            keep = cls == 0
            boxes = [bb.astype(np.float32) for bb, k in zip(xyxy, keep) if k]

            feats: List[torch.Tensor] = []
            for bb in boxes:
                x1, y1, x2, y2 = map(int, bb)
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                feats.append(extractor(crop))

            out_queue.put(("data", cam_id, ts, frame, boxes, feats))

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out_queue.put(("eof", cam_id))
        print(f"[Cam {cam_id}] finished.")

def fusion_process(num_cams: int,
                   in_queue: Queue,
                   result_queue: Queue,
                   stop_event: Event,
                   show: bool = True) -> None:
    assigner = GlobalIDAssigner(dist_th=0.75, alpha=0.95, max_misses=150)
    tmp.set_sharing_strategy("file_system")
    # Per-camera tracked store keyed by gid
    tracked_by_cam: List[Dict[int, List[Tuple[str, torch.Tensor]]]] = [dict() for _ in range(num_cams)]

    open_cams = set(range(num_cams))

    while open_cams and not stop_event.is_set():
        try:
            msg = in_queue.get(timeout=1.0)
        except Exception:
            continue

        kind = msg[0]
        if kind == "eof":
            _, cam_id = msg
            open_cams.discard(cam_id)
            continue

        _, cam_id, ts, frame, boxes, feats = msg

        # Assign global IDs for this batch of detections from one camera
        dets = [(cam_id, np.asarray(bb, dtype=np.float32), fv) for bb, fv in zip(boxes, feats)]
        gids = assigner.update(dets, ts)

        # Update per-camera tracked store
        for (bb, fv, gid) in zip(boxes, feats, gids):
            lst = tracked_by_cam[cam_id].setdefault(gid, [])
            lst.append((f"{ts:.2f}", fv))

        # Optional drawing (one window per camera)
        if show:
            vis = frame.copy()
            for (bb, gid) in zip(boxes, gids):
                x1, y1, x2, y2 = map(int, bb)
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(vis, f"ID: {str(gid)}", (x1, max(0, y1 - 6)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            resized_vis = cv.resize(vis, new_dimensions, interpolation=cv.INTER_AREA)
            cv.imshow(f"Cam {cam_id}", resized_vis)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    # Close windows
    if show:
        for cam in range(num_cams):
            try:
                cv.destroyWindow(f"Cam {cam}")
            except Exception:
                pass

    serial = []
    for cam_dict in tracked_by_cam:
        sdict = {}
        for gid, pairs in cam_dict.items():
            sdict[gid] = [(ts, feat.detach().cpu().numpy().astype(np.float32))
                        for ts, feat in pairs]
        serial.append(sdict)

    result_queue.put(serial)            # send NumPy arrays, not torch.Tensors
    # Send results back
    #result_queue.put(tracked_by_cam)

# =============================================================================
# Entrypoint
# =============================================================================

def main(video_paths, cfg_file, weight_file, show = True):
    tmp.set_sharing_strategy("file_system")
    set_start_method("fork", force=True)

    stop_event = mp.Event()

    # One queue that all camera workers push into; fusion consumes
    data_queue: Queue = mp.Queue(maxsize=128)
    result_queue: Queue = mp.Queue()

    cam_procs: List[Process] = []
    for cam_id, path in enumerate(video_paths):
        p = Process(target=camera_worker,
                    args=(cam_id, path, cfg_file, weight_file, data_queue, stop_event),
                    daemon=True)
        p.start()
        cam_procs.append(p)

    fuse = Process(target=fusion_process,
                   args=(len(video_paths), data_queue, result_queue, stop_event, show),
                   daemon=False)
    fuse.start()

    try:
        tracked_by_cam = result_queue.get()  # blocks until fusion finishes
    except KeyboardInterrupt:
        print("KeyboardInterrupt - stopping …")
        stop_event.set()
        tracked_by_cam = [{} for _ in video_paths]
    finally:
        for p in cam_procs:
            p.join()
        fuse.join()

    # --- Summary -------------------------------------------------------------
    print("\n=== Summary ===")
    for cam_id, d in enumerate(tracked_by_cam):
        print(f"Camera {cam_id}: {len(d)} unique global IDs")

    # Optionally, build the (timestamp, tensor) lists like before
    cam_feats: List[List[Tuple[str, torch.Tensor]]] = []
    for d in tracked_by_cam:
        lst: List[Tuple[str, torch.Tensor]] = []
        for gid, pairs in d.items():
            lst.extend(pairs)
        cam_feats.append(lst)

    #PCA Plots
    '''all_vecs, gid_labels, cam_labels = [], [], []
    for cam_id, cam_dict in enumerate(tracked_by_cam):
        for gid, pairs in cam_dict.items():
            for ts, feat in pairs:
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                all_vecs.append(feat)
                gid_labels.append(gid)
                cam_labels.append(cam_id)

    if not all_vecs:
        print("No feature vectors available – skipping PCA plots.")
        exit()

    X = np.vstack(all_vecs)
    X2 = PCA(n_components=2, random_state=0).fit_transform(X)   # (N, 2)

    gid_labels = np.asarray(gid_labels)
    cam_labels = np.asarray(cam_labels)

    # ------------------------------------------------------------------
    # 2.  Prepare colour maps
    # ------------------------------------------------------------------
    unique_gids  = np.unique(gid_labels)
    cmap_gid     = cm.get_cmap("tab20", len(unique_gids))      #  ≤ 20 hues
    color_gid    = {gid: cmap_gid(i) for i, gid in enumerate(unique_gids)}

    colors_cam   = ["blue", "red", "seagreen", "orchid", "gold", "magenta"]
    n_cams       = len(video_paths)

    # ------------------------------------------------------------------
    # 3.  Two‑panel figure
    # ------------------------------------------------------------------
    fig, (ax_id, ax_cam) = plt.subplots(
        1, 2, figsize=(13, 6), constrained_layout=True, sharex=True, sharey=True
    )

    # --- left: colour by GLOBAL ID ------------------------------------
    for gid in unique_gids:
        idx = gid_labels == gid
        ax_id.scatter(
            X2[idx, 0], X2[idx, 1],
            s=22, alpha=0.8,
            c=[color_gid[gid]],
            edgecolors="none",
            label=f"ID {gid}",
        )
    ax_id.set_title("PCA – colour‑coded by global ID")
    ax_id.set_xlabel("PC 1"); ax_id.set_ylabel("PC 2")
    ax_id.legend(title="Global IDs", loc="best", fontsize="small")

    # --- right: colour by CAMERA ID -----------------------------------
    for cam_id in range(n_cams):
        idx = cam_labels == cam_id
        if idx.any():
            ax_cam.scatter(
                X2[idx, 0], X2[idx, 1],
                s=22, alpha=0.8,
                c=[colors_cam[cam_id % len(colors_cam)]],
                edgecolors="none",
                label=f"Cam {cam_id}",
            )
    ax_cam.set_title("PCA – colour‑coded by camera")
    ax_cam.set_xlabel("PC 1"); ax_cam.set_ylabel("PC 2")
    ax_cam.legend(title="Cameras", loc="best", fontsize="small")

    plt.show()'''
    
    #3D pca
    all_vecs, labels = [], []
    for cam_id, cam_dict in enumerate(tracked_by_cam):
        for gid, pairs in cam_dict.items():
            for ts, feat in pairs:
                all_vecs.append((feat))
                labels.append(cam_id)
    if all_vecs:
        X = np.vstack(all_vecs)                    # (N, D)
        pca = PCA(n_components=3, random_state=0)  # 3 components for 3‑D plot
        X3 = pca.fit_transform(X)                  # (N, 3)

        colors = ["blue", "red", "seagreen", "orchid", "gold", "magenta"]
        labels_arr = np.array(labels)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        for cam_id in range(len(video_paths)):
            idx = labels_arr == cam_id
            if idx.any():
                ax.scatter(
                    X3[idx, 0], X3[idx, 1], X3[idx, 2],
                    s=22, alpha=0.8,
                    c=colors[cam_id % len(colors)],
                    label=f"Cam {cam_id}",
                    edgecolors="none",
                )

    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    ax.set_zlabel("PC-3")
    ax.set_title("3-D PCA - global IDs across cameras")
    ax.legend()
    plt.tight_layout()
    plt.show()


    all_vecs   : list[np.ndarray] = []
    gid_labels : list[int] = []

    for cam_dict in tracked_by_cam:                 # one dict per camera
        for gid, pairs in cam_dict.items():         # gid = global ID
            for ts, feat in pairs:                  # (timestamp, feature)
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                all_vecs.append(feat)
                gid_labels.append(gid)

    # ------------------------------------------------------------------
    # 2.  3‑D PCA, then colour‑code by ID
    # ------------------------------------------------------------------
    if all_vecs:
        X  = np.vstack(all_vecs)                    # (N, D)
        X3 = PCA(n_components=3, random_state=0).fit_transform(X)

        gid_labels = np.asarray(gid_labels)
        unique_gids = np.unique(gid_labels)

        cmap = cm.get_cmap("tab20", len(unique_gids))   # distinct hue per ID
        gid2color = {gid: cmap(i) for i, gid in enumerate(unique_gids)}

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")

        for gid in unique_gids:
            idx = gid_labels == gid
            ax.scatter(
                X3[idx, 0], X3[idx, 1], X3[idx, 2],
                s=24, alpha=0.85,
                c=[gid2color[gid]],
                edgecolors="none",
                label=f"ID {gid}",
            )

        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.set_zlabel("PC-3")
        ax.set_title("3-D PCA - colour-coded by global ID")
        ax.legend(title="Global IDs")
        plt.tight_layout()
        plt.show()
    else:
        print("No feature vectors available - skipping 3-D PCA plot.")
    
    


'''RUN IN TERMINAL WITH python3 -m mobreid.multicam_test '''
if __name__ == "__main__":
    tmp.set_sharing_strategy("file_system")

    video_paths1 = [
        HERE / "reu_1person_cam0.mp4",
        HERE / "reu_1person_cam1.mp4",
        HERE / "reu_1person_cam2.mp4",
        HERE / "reu_1person_cam3_edit.mp4"
    ]

    video_paths3 = [
        HERE / "reu_3person_cam0.mp4",
        HERE / "reu_3person_cam1.mp4",
        HERE / "reu_3person_cam2.mp4",
        HERE / "reu_3person_cam3_edit.mp4"
    ]

    cfg_file = "logs/market1501/mgn_R50-ibn/config.yaml"  # path to model config
    weight_file = "logs/market1501/mgn_R50-ibn/model_final.pth"  # path to pretrained model
    main(video_paths3, cfg_file, weight_file, show=True)
