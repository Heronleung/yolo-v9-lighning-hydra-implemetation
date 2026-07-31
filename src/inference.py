"""src/inference.py

Hydra entry point for YOLOv9 inference (M7).

Behaviour mirrors upstream detect.py / detect_dual.py one-to-one (decision
2026-07-09): same defaults, same runs/detect/exp{n} output layout, same
labels/ save-txt format, same drawing and per-image console summary. Hydra
only replaces the argparse shell -- the pixel-in / boxes-out path reuses the
upstream building blocks untouched:

  LoadImages (letterbox inside) -> model forward -> non_max_suppression
  -> scale_boxes -> Annotator -> cv2.imwrite

Scope (M7): images / folders / globs only. Video, webcam, stream sources and
ONNX / TensorRT backends are deferred.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the repo root is importable so the TOP-LEVEL `models` / `utils`
# packages resolve to the upstream YOLOv9 code (same trick as src/train.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, increment_path,
                           non_max_suppression, scale_boxes, strip_optimizer,
                           xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Hydra's job_logging captures the logging module (NOT print), so everything
# below also lands in inference.log (train.log lesson from 2026-07-09).
log = logging.getLogger(__name__)


def select_inference_output(pred):
    """Mirror the upstream branch selection, auto-detecting the ckpt family.

    detect.py (converted single-branch DDetect): the raw model output goes
    straight to NMS (non_max_suppression itself unwraps a (y, x) tuple).
    detect_dual.py (dual-branch DualDDetect): `pred = pred[0][1]` -- the
    inference output is a pair [aux, lead]; keep the LEAD branch, exactly as
    src/lit_module.py validation already does.
    """
    if isinstance(pred, (list, tuple)) and len(pred) > 0 and isinstance(pred[0], (list, tuple)):
        return pred[0][1]
    return pred


@smart_inference_mode()
def run_inference(cfg: DictConfig) -> None:
    source = str(cfg.source)
    save_img = not bool(cfg.nosave) and not source.endswith(".txt")

    # Output dir -- upstream layout: runs/detect/exp, exp2, ... (+ labels/).
    project = Path(str(cfg.project))
    if not project.is_absolute():
        project = REPO_ROOT / project  # keep runs anchored at the repo root
    save_dir = increment_path(project / str(cfg.name), exist_ok=bool(cfg.exist_ok))
    (save_dir / "labels" if cfg.save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Model. DetectMultiBackend -> attempt_load prefers ckpt["ema"] over
    # ckpt["model"], so Lightning last.pt/best.pt AND M6 converted ckpts load
    # with no extra flag.
    device = select_device(str(cfg.device))
    # `backend_data` is the Hydra equivalent of upstream `--data`. The name
    # cannot be `data` because that key is already the Hydra data-config group.
    backend_data = str(cfg.backend_data) if cfg.get("backend_data") else None
    model = DetectMultiBackend(
        str(cfg.weights), device=device, dnn=bool(cfg.dnn),
        data=backend_data, fp16=bool(cfg.half),
    )
    stride, names, pt = model.stride, model.names, model.pt
    if cfg.data.get("names"):
        names = list(cfg.data.names)  # single source of truth: configs/data (18 TACO classes)
    imgsz = check_img_size([int(cfg.imgsz), int(cfg.imgsz)], s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=int(cfg.vid_stride))
    classes = [int(c) for c in cfg.classes] if cfg.classes is not None else None

    model.warmup(imgsz=(1, 3, *imgsz))
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        if vid_cap is not None:
            raise NotImplementedError("M7 scope: images/folders only -- video/stream inference is deferred.")

        with dt[0]:  # pre-process (letterbox already applied inside LoadImages)
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:  # forward
            # Mirrors detect_dual.py --visualize: create one feature directory
            # per input image and pass it into the model forward call.
            visualize_dir = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if bool(cfg.visualize) else False
            )
            pred = model(
                im,
                augment=bool(cfg.augment),
                visualize=visualize_dir,
            )
            pred = select_inference_output(pred)

        with dt[2]:  # NMS
            pred = non_max_suppression(
                pred, float(cfg.conf_thres), float(cfg.iou_thres),
                classes, bool(cfg.agnostic_nms), max_det=int(cfg.max_det),
            )

        det = pred[0]  # batch size 1 (LoadImages)
        seen += 1
        p, im0 = Path(path), im0s.copy()
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / "labels" / p.stem)
        s += "%gx%g " % im.shape[2:]
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # whwh normalization gain
        imc = im0.copy() if cfg.save_crop else im0
        annotator = Annotator(im0, line_width=int(cfg.line_thickness), example=str(names))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, 5].unique():  # per-image console summary, upstream format
                n = (det[:, 5] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in reversed(det):
                if cfg.save_txt:  # upstream YOLO label format (+ conf optional)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if cfg.save_conf else (cls, *xywh)
                    with open(f"{txt_path}.txt", "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")
                if save_img or cfg.save_crop or cfg.view_img:
                    c = int(cls)
                    label = None if cfg.hide_labels else (names[c] if cfg.hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if cfg.save_crop:
                    save_one_box(xyxy, imc, file=save_dir / "crops" / names[int(cls)] / f"{p.stem}.jpg", BGR=True)
        im0 = annotator.result()
        if cfg.view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)
        if save_img:
            cv2.imwrite(save_path, im0)
        log.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    if seen:
        t = tuple(x.t / seen * 1e3 for x in dt)
        log.info("Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape %s"
                 % (*t, (1, 3, *imgsz)))
    if cfg.save_txt or save_img:
        log.info(f"Results saved to {save_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    # Fail loudly on unresolved ${...}; log the resolved config (same as train.py).
    log.info("=" * 80)
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("=" * 80)
    run_inference(cfg)

    # Mirrors detect_dual.py --update. This intentionally mutates the source
    # checkpoint by stripping optimizer state after successful inference.
    if bool(cfg.update):
        log.info("[update] stripping optimizer from %s", cfg.weights)
        strip_optimizer(str(cfg.weights))


if __name__ == "__main__":
    main()