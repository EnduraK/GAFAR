#!/usr/bin/env python3
"""Measure verification EER for full-face vs periocular crops on a real dataset.

Runs the whole pipeline — image -> MediaPipe landmarks -> crop -> embedding -> EER
— for two regions (whole image and the periocular crop) so you can see the
upper-face-only penalty directly.

Public baseline (no consent needed): Labelled Faces in the Wild (LFW), fetched via
scikit-learn. Your real target is your own 20-volunteer masked/unmasked set — point
--images at a folder of `identity/xxx.jpg` and it works the same way.

    pip install facenet-pytorch scikit-image scikit-learn
    python scripts/eval_periocular.py --max-images 400

Backends: facenet (recommended, real face model, MIT) or dlib.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def load_lfw(min_faces, max_images):
    from sklearn.datasets import fetch_lfw_people
    data = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=0.5)
    imgs = (data.images * 255).astype(np.uint8)     # (N,H,W,3) RGB 0..1 -> 0..255
    labels = data.target
    if max_images and len(imgs) > max_images:
        idx = np.linspace(0, len(imgs) - 1, max_images).astype(int)
        imgs, labels = imgs[idx], labels[idx]
    return [np.ascontiguousarray(i[:, :, ::-1]) for i in imgs], list(labels)   # -> BGR


def load_folder(root, max_images):
    import cv2
    root = Path(root)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    imgs, labels = [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(img); labels.append(p.parent.name)
    if max_images and len(imgs) > max_images:
        idx = np.linspace(0, len(imgs) - 1, max_images).astype(int)
        imgs = [imgs[i] for i in idx]; labels = [labels[i] for i in idx]
    return imgs, labels


def build_embedder(name):
    if name == "facenet":
        from cranium.embedding import FacenetEmbedder
        return FacenetEmbedder()
    if name == "dlib":
        import os
        sp = os.environ.get("DLIB_SHAPE"); rec = os.environ.get("DLIB_REC")
        if not (sp and rec):
            sys.exit("dlib backend needs DLIB_SHAPE and DLIB_REC env vars (model paths)")
        from cranium.embedding import DlibEmbedder
        return DlibEmbedder(sp, rec)
    sys.exit(f"unknown backend {name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lfw", "folder"], default="lfw")
    ap.add_argument("--images", help="folder of identity/xxx.jpg (for --dataset folder)")
    ap.add_argument("--min-faces", type=int, default=20, help="LFW: min images per identity")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--backend", choices=["facenet", "dlib"], default="facenet")
    ap.add_argument("--out", default="docs/eer_result.png")
    args = ap.parse_args(argv)

    from cranium.landmarks import FaceMeshLandmarker
    from cranium.periocular import crop_periocular
    from cranium.evaluation import evaluate, pairwise_scores, compute_eer

    print(f"[1/4] loading dataset ({args.dataset})…")
    imgs, labels = (load_folder(args.images, args.max_images) if args.dataset == "folder"
                    else load_lfw(args.min_faces, args.max_images))
    print(f"      {len(imgs)} images, {len(set(labels))} identities")

    print(f"[2/4] building embedder ({args.backend})…")
    emb = build_embedder(args.backend)

    print("[3/4] landmarking + embedding (full face vs periocular)…")
    full_e, peri_e, keep = [], [], []
    t0 = time.time()
    with FaceMeshLandmarker() as lm:
        for k, img in enumerate(imgs):
            pts = lm.detect(img)
            if pts is None:
                continue
            crop, _ = crop_periocular(img, pts)
            if crop.size == 0:
                continue
            try:
                full_e.append(emb.embed(img))
                peri_e.append(emb.embed(crop))
                keep.append(labels[k])
            except Exception:
                continue
            if (k + 1) % 50 == 0:
                print(f"      {k+1}/{len(imgs)}  ({time.time()-t0:.0f}s)")
    print(f"      usable: {len(keep)} images across {len(set(keep))} identities")
    if len(set(keep)) < 2:
        sys.exit("need at least two identities with detected faces")

    print("[4/4] computing EER…\n")
    full_r = evaluate(full_e, keep)
    peri_r = evaluate(peri_e, keep)
    print("FULL FACE   :", full_r.summary())
    print("PERIOCULAR  :", peri_r.summary())
    penalty = peri_r.eer / full_r.eer if full_r.eer > 0 else float("inf")
    print(f"\nPeriocular EER is {penalty:.1f}x the full-face EER "
          f"({peri_r.eer*100:.2f}% vs {full_r.eer*100:.2f}%).")
    print(f"Phase 1 gate (EER < 3%): periocular {'PASS' if peri_r.eer < 0.03 else 'NOT YET'}.")

    try:
        _plot(peri_e, keep, peri_r, args.out)
        print(f"\nSaved score-distribution figure -> {args.out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


def _plot(embs, labels, result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cranium.evaluation import pairwise_scores
    g, i = pairwise_scores(embs, labels)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(i, bins=60, alpha=.6, label=f"impostor (n={len(i)})", color="#fb7185", density=True)
    ax.hist(g, bins=60, alpha=.6, label=f"genuine (n={len(g)})", color="#34d399", density=True)
    ax.axvline(result.threshold, color="#fbbf24", ls="--", label=f"EER threshold {result.threshold:.2f}")
    ax.set_title(f"Periocular verification — EER {result.eer*100:.2f}%  (AUC {result.auc:.3f})")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density"); ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
