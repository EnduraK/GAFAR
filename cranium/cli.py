"""CRANIUM command-line interface.

  enrol      capture one or more samples and store a template for a subject
  verify     1:1 verify a probe against an enrolled subject
  authorise  run a full authorisation event (verify + policy + optional liveness)

Input is either --image (repeatable, headless-friendly) or --webcam.

Embedding backends:
  stub   deterministic, model-free — exercises the pipeline; NOT a recogniser
  dlib   real 128-D descriptor (Boost licence). Needs the two dlib model files:
         --dlib-shape shape_predictor_5_face_landmarks.dat
         --dlib-rec   dlib_face_recognition_resnet_model_v1.dat
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

from .landmarks import FaceMeshLandmarker
from .embedding import StubEmbedder, DlibEmbedder
from .pipeline import PeriocularPipeline
from .verifier import Verifier, TemplateStore
from .authorization import AuthorizationEngine, AuthorizationRequest
from .liveness import BlinkChallenge, eye_aspect_ratios


def _build_embedder(args):
    if args.backend == "dlib":
        if not (args.dlib_shape and args.dlib_rec):
            sys.exit("dlib backend needs --dlib-shape and --dlib-rec")
        return DlibEmbedder(args.dlib_shape, args.dlib_rec)
    print("[warn] using StubEmbedder: wiring only, not a real recogniser", file=sys.stderr)
    return StubEmbedder()


def _load_image(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"cannot read image: {path}")
    return img


def _embed_images(pipe, paths):
    out = []
    for p in paths:
        res = pipe.embed(_load_image(p))
        if res is None:
            print(f"[warn] no face found in {p}", file=sys.stderr)
            continue
        out.append(res.embedding)
    return out


def _capture_webcam(pipe, n_frames=1, show_ear=False):
    """Grab n usable frames from the default camera. Returns list of embeddings."""
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("cannot open webcam (device 0)")
    got = []
    print(f"[info] capturing {n_frames} sample(s) — look at the camera…", file=sys.stderr)
    tries = 0
    while len(got) < n_frames and tries < 300:
        tries += 1
        ok, frame = cap.read()
        if not ok:
            continue
        res = pipe.embed(frame)
        if res is not None:
            got.append(res.embedding)
    cap.release()
    if not got:
        sys.exit("no face captured from webcam")
    return got


def _store_path(args):
    return Path(args.store)


def _load_store(args):
    p = _store_path(args)
    return TemplateStore.load(p) if p.exists() else TemplateStore()


def cmd_enrol(args):
    with FaceMeshLandmarker() as lm:
        pipe = PeriocularPipeline(lm, _build_embedder(args))
        embs = (_capture_webcam(pipe, n_frames=args.samples) if args.webcam
                else _embed_images(pipe, args.image))
    if not embs:
        sys.exit("no usable samples")
    store = _load_store(args)
    v = Verifier(store)
    tmpl = v.enroll(args.subject, embs, meta={"backend": args.backend})
    store.save(_store_path(args))
    print(f"enrolled '{args.subject}' from {tmpl.n_samples} sample(s) -> {args.store}")


def cmd_verify(args):
    store = _load_store(args)
    if not store.has(args.subject):
        sys.exit(f"subject '{args.subject}' is not enrolled in {args.store}")
    with FaceMeshLandmarker() as lm:
        pipe = PeriocularPipeline(lm, _build_embedder(args))
        probes = (_capture_webcam(pipe, 1) if args.webcam
                  else _embed_images(pipe, args.image))
    if not probes:
        sys.exit("no probe captured")
    v = Verifier(store, threshold=args.threshold)
    res = v.verify(args.subject, probes[0])
    verdict = "MATCH" if res.is_match else "NO MATCH"
    print(f"{verdict}  score={res.score:.3f}  threshold={res.threshold:.3f}  subject={args.subject}")
    sys.exit(0 if res.is_match else 2)


def cmd_authorise(args):
    store = _load_store(args)
    with FaceMeshLandmarker() as lm:
        pipe = PeriocularPipeline(lm, _build_embedder(args))
        probes = (_capture_webcam(pipe, 1) if args.webcam
                  else _embed_images(pipe, args.image))
    if not probes:
        sys.exit("no probe captured")

    engine = AuthorizationEngine(Verifier(store))
    liveness_passed = args.liveness  # --image path: pass explicitly; webcam does a live blink below

    if args.webcam and engine.policy_for(args.risk).require_liveness:
        liveness_passed = _webcam_blink()

    req = AuthorizationRequest(action=args.action, claimed_subject_id=args.subject,
                               risk_level=args.risk, resource=args.resource)
    decision = engine.authorize(req, probes[0], liveness_passed=liveness_passed)
    print(f"\n{'AUTHORISED' if decision.approved else 'DENIED'} — {args.action} — {args.resource}")
    print(f"  score {decision.score:.3f} vs threshold {decision.threshold:.3f} "
          f"(risk={args.risk}, liveness_required={decision.liveness_required}, "
          f"liveness_passed={decision.liveness_passed})")
    for r in decision.reasons:
        print(f"  - {r}")
    print(f"  audit hash: {engine.audit.records[-1]['hash'][:24]}…  "
          f"chain_ok={engine.audit.verify_chain()}")
    sys.exit(0 if decision.approved else 2)


def _webcam_blink(window_s=5.0):
    import cv2
    import time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    print("[info] liveness: please BLINK once…", file=sys.stderr)
    ch = BlinkChallenge(required_blinks=1, window_s=window_s)
    with FaceMeshLandmarker() as lm:
        t0 = time.time()
        while time.time() - t0 < window_s + 1:
            ok, frame = cap.read()
            if not ok:
                continue
            pts = lm.detect(frame)
            if pts is None:
                continue
            _, _, ear = eye_aspect_ratios(pts)
            r = ch.update(ear, time.time())
            if r.passed:
                cap.release()
                print("[info] liveness confirmed", file=sys.stderr)
                return True
    cap.release()
    print("[info] liveness NOT confirmed", file=sys.stderr)
    return False


def build_parser():
    p = argparse.ArgumentParser(prog="cranium", description="Upper-face verification & authorisation (MVP)")
    p.add_argument("--store", default="templates.json", help="template store JSON path")
    p.add_argument("--backend", choices=["stub", "dlib"], default="stub")
    p.add_argument("--dlib-shape", help="dlib shape predictor .dat")
    p.add_argument("--dlib-rec", help="dlib recognition model .dat")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_input(sp):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--image", nargs="+", help="one or more image paths")
        g.add_argument("--webcam", action="store_true", help="capture from camera 0")

    e = sub.add_parser("enrol", help="enrol a subject")
    e.add_argument("--subject", required=True)
    e.add_argument("--samples", type=int, default=5, help="webcam frames to average")
    add_input(e); e.set_defaults(func=cmd_enrol)

    v = sub.add_parser("verify", help="1:1 verify")
    v.add_argument("--subject", required=True)
    v.add_argument("--threshold", type=float, default=0.62)
    add_input(v); v.set_defaults(func=cmd_verify)

    a = sub.add_parser("authorise", help="run an authorisation event")
    a.add_argument("--subject", required=True)
    a.add_argument("--action", default="release_medication")
    a.add_argument("--resource", default="")
    a.add_argument("--risk", choices=["low", "standard", "high", "critical"], default="standard")
    a.add_argument("--liveness", action="store_true",
                   help="(image mode) assert liveness passed; webcam mode runs a live blink check")
    add_input(a); a.set_defaults(func=cmd_authorise)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
