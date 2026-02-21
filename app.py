"""
Face Clustering Photo Organizer

Scans a folder of photos, detects faces, clusters them by person,
and lets you browse person-by-person to select which to keep.
Selected photos are copied to an output folder.

Run:
    uvicorn app:app --reload
"""

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
CLUSTER_THRESHOLD = 0.4
CACHE_FILENAME = ".face-cluster-cache.json"
CACHE_VERSION = 1

# ---------------------------------------------------------------------------
# Face detector (loaded once)
# ---------------------------------------------------------------------------

_detector = None
_detector_lock = threading.Lock()


def _get_detector():
    """Lazy-load the face detector on first use."""
    global _detector
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is not None:
            return _detector
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _detector = app
        return _detector


def detect_faces(image_path: str):
    """Return list of detected faces, or empty list on read error."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    return _get_detector().get(img)


# ---------------------------------------------------------------------------
# Clustering data structures
# ---------------------------------------------------------------------------


@dataclass
class PersonCluster:
    id: int
    centroid: np.ndarray
    image_filenames: list[str] = field(default_factory=list)
    image_similarities: dict[str, float] = field(default_factory=dict)
    count: int = 0  # number of embeddings merged into centroid


@dataclass
class ScanState:
    scan_folder: str = ""
    output_folder: str = ""
    refs_folder: str = ""
    status: str = "idle"  # idle | running | done | error
    processed: int = 0
    total: int = 0
    clusters: list[PersonCluster] = field(default_factory=list)
    no_face_images: list[str] = field(default_factory=list)
    image_face_counts: dict[str, int] = field(default_factory=dict)
    ref_matches: dict[int, float] = field(default_factory=dict)  # cluster_id → best similarity
    error: str = ""


scan_state = ScanState()
_scan_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_images(folder: Path) -> list[Path]:
    """Recursively collect all image files under *folder*."""
    return sorted(
        f
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _cosine_similarity(a, b) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


REF_SIMILARITY_THRESHOLD = 0.3


def _compute_ref_matches(state: ScanState) -> dict[int, float]:
    """Score each cluster centroid against reference face embeddings.

    Returns {cluster_id: best_similarity} for clusters above threshold.
    """
    if not state.refs_folder:
        return {}

    refs_path = Path(state.refs_folder)
    if not refs_path.is_dir():
        return {}

    ref_images = _collect_images(refs_path)
    if not ref_images:
        return {}

    # Extract embeddings from reference photos
    ref_embeddings = []
    for img_path in ref_images:
        faces = detect_faces(str(img_path))
        for face in faces:
            ref_embeddings.append(face.normed_embedding)

    if not ref_embeddings:
        return {}

    # Score each cluster centroid against all ref embeddings
    matches: dict[int, float] = {}
    for cluster in state.clusters:
        best_sim = max(
            _cosine_similarity(cluster.centroid, ref_emb)
            for ref_emb in ref_embeddings
        )
        if best_sim >= REF_SIMILARITY_THRESHOLD:
            matches[cluster.id] = best_sim

    return matches


# ---------------------------------------------------------------------------
# Cache save / load
# ---------------------------------------------------------------------------


def _cache_path(scan_folder: str) -> Path:
    return Path(scan_folder) / CACHE_FILENAME


def _save_cache(state: ScanState):
    """Persist clustering results to a JSON file in the scan folder."""
    data = {
        "version": CACHE_VERSION,
        "scan_folder": state.scan_folder,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "image_count": state.total,
        "clusters": [
            {
                "id": c.id,
                "centroid": c.centroid.tolist(),
                "image_filenames": c.image_filenames,
                "image_similarities": c.image_similarities,
                "count": c.count,
            }
            for c in state.clusters
        ],
        "no_face_images": state.no_face_images,
        "image_face_counts": state.image_face_counts,
    }
    path = _cache_path(state.scan_folder)
    path.write_text(json.dumps(data, indent=2))


def _load_cache(scan_folder: str) -> dict | None:
    """Load cached results if they exist, return None otherwise."""
    path = _cache_path(scan_folder)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != CACHE_VERSION:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_to_state(cache: dict, output_folder: str) -> ScanState:
    """Rebuild ScanState from cached JSON data."""
    clusters = []
    for c in cache["clusters"]:
        clusters.append(
            PersonCluster(
                id=c["id"],
                centroid=np.array(c["centroid"], dtype=np.float32),
                image_filenames=c["image_filenames"],
                image_similarities=c["image_similarities"],
                count=c["count"],
            )
        )
    return ScanState(
        scan_folder=cache["scan_folder"],
        output_folder=output_folder,
        status="done",
        processed=cache["image_count"],
        total=cache["image_count"],
        clusters=clusters,
        no_face_images=cache["no_face_images"],
        image_face_counts=cache["image_face_counts"],
    )


# ---------------------------------------------------------------------------
# Background scanning with clustering
# ---------------------------------------------------------------------------


def _run_scan(scan_folder: str):
    global scan_state
    folder_path = Path(scan_folder)

    image_files = _collect_images(folder_path)
    scan_state.total = len(image_files)

    clusters: list[PersonCluster] = []
    no_face_images: list[str] = []
    image_face_counts: dict[str, int] = {}
    next_cluster_id = 0

    for img_path in image_files:
        rel = str(img_path.relative_to(folder_path))
        faces = detect_faces(str(img_path))
        image_face_counts[rel] = len(faces)

        if not faces:
            no_face_images.append(rel)
            scan_state.processed += 1
            continue

        for face in faces:
            embedding = face.normed_embedding

            best_cluster = None
            best_sim = 0.0

            for cluster in clusters:
                sim = _cosine_similarity(embedding, cluster.centroid)
                if sim > CLUSTER_THRESHOLD and sim > best_sim:
                    best_cluster = cluster
                    best_sim = sim

            if best_cluster is not None:
                if rel not in best_cluster.image_filenames:
                    best_cluster.image_filenames.append(rel)
                prev_sim = best_cluster.image_similarities.get(rel, 0.0)
                if best_sim > prev_sim:
                    best_cluster.image_similarities[rel] = best_sim
                # Running average centroid update
                n = best_cluster.count
                best_cluster.centroid = (best_cluster.centroid * n + embedding) / (
                    n + 1
                )
                best_cluster.count += 1
            else:
                clusters.append(
                    PersonCluster(
                        id=next_cluster_id,
                        centroid=embedding.copy(),
                        image_filenames=[rel],
                        image_similarities={rel: 1.0},
                        count=1,
                    )
                )
                next_cluster_id += 1

        scan_state.processed += 1

    # Sort clusters by number of images (largest first)
    clusters.sort(key=lambda c: len(c.image_filenames), reverse=True)

    scan_state.clusters = clusters
    scan_state.no_face_images = no_face_images
    scan_state.image_face_counts = image_face_counts
    scan_state.ref_matches = _compute_ref_matches(scan_state)
    scan_state.status = "done"

    _save_cache(scan_state)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Face Clustering Photo Organizer")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/browse")
def browse_folders(path: str = Query(default="")):
    """List subdirectories and image count for a given path."""
    if not path:
        path = str(Path.home())

    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    dirs = []
    image_count = 0
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    return {
        "path": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "dirs": dirs,
        "image_count": image_count,
    }


# --- Request / Response models ---


class ScanRequest(BaseModel):
    scan_folder: str
    output_folder: str
    refs_folder: str = ""


class CacheLoadRequest(BaseModel):
    scan_folder: str
    output_folder: str
    refs_folder: str = ""


class ApplyRequest(BaseModel):
    selected: list[str]


# --- Endpoints ---


@app.get("/api/cache/check")
def check_cache(scan_folder: str = Query()):
    """Check whether a scan cache exists for the given folder."""
    cache = _load_cache(scan_folder)
    if cache is None:
        return {"has_cache": False}
    return {
        "has_cache": True,
        "scanned_at": cache.get("scanned_at", ""),
        "image_count": cache.get("image_count", 0),
    }


@app.post("/api/cache/load")
def load_cache(req: CacheLoadRequest):
    """Load previous scan results from cache instead of re-scanning."""
    global scan_state

    cache = _load_cache(req.scan_folder)
    if cache is None:
        raise HTTPException(status_code=404, detail="No cache found for this folder")

    with _scan_lock:
        if scan_state.status == "running":
            raise HTTPException(status_code=409, detail="A scan is already running")
        scan_state = _cache_to_state(cache, req.output_folder)
        scan_state.refs_folder = req.refs_folder
        scan_state.ref_matches = _compute_ref_matches(scan_state)

    return {
        "message": "Loaded previous scan results",
        "image_count": cache["image_count"],
    }


@app.post("/api/scan")
def start_scan(req: ScanRequest):
    global scan_state

    scan_path = Path(req.scan_folder)
    if not scan_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Scan folder not found: {req.scan_folder}"
        )

    output_path = Path(req.output_folder)
    if not output_path.is_dir():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Cannot create output folder: {e}"
            )

    with _scan_lock:
        if scan_state.status == "running":
            raise HTTPException(status_code=409, detail="A scan is already running")

        image_count = len(_collect_images(scan_path))

        scan_state = ScanState(
            scan_folder=req.scan_folder,
            output_folder=req.output_folder,
            refs_folder=req.refs_folder,
            status="running",
            total=image_count,
        )

    thread = threading.Thread(target=_run_scan, args=(req.scan_folder,), daemon=True)
    thread.start()

    return {"message": "Scan started", "scan_folder": req.scan_folder, "total": image_count}


@app.get("/api/scan/status")
def scan_status():
    return {
        "status": scan_state.status,
        "processed": scan_state.processed,
        "total": scan_state.total,
    }


@app.get("/api/results")
def get_results():
    if scan_state.status not in ("done", "running"):
        raise HTTPException(status_code=400, detail="No scan has been run yet")

    has_refs = bool(scan_state.refs_folder and scan_state.ref_matches)

    # Deduplicate: assign each image to the cluster where it has the
    # highest similarity so group photos don't repeat across clusters.
    best_cluster_for: dict[str, tuple[int, float]] = {}  # filename → (cluster_id, sim)
    for cluster in scan_state.clusters:
        for filename in cluster.image_filenames:
            sim = cluster.image_similarities.get(filename, 0.0)
            prev = best_cluster_for.get(filename)
            if prev is None or sim > prev[1]:
                best_cluster_for[filename] = (cluster.id, sim)

    persons = []
    for cluster in scan_state.clusters:
        images = []
        for filename in cluster.image_filenames:
            if best_cluster_for.get(filename, (None,))[0] != cluster.id:
                continue
            images.append(
                {
                    "filename": filename,
                    "face_count": scan_state.image_face_counts.get(filename, 0),
                    "similarity": round(
                        cluster.image_similarities.get(filename, 0.0), 3
                    ),
                }
            )
        if not images:
            continue
        ref_sim = scan_state.ref_matches.get(cluster.id, 0.0)
        persons.append(
            {
                "id": cluster.id,
                "label": f"Person {cluster.id + 1}",
                "photo_count": len(images),
                "matched": cluster.id in scan_state.ref_matches,
                "ref_similarity": round(ref_sim, 3),
                "images": images,
            }
        )

    if scan_state.no_face_images:
        no_face_imgs = [
            {"filename": f, "face_count": 0, "similarity": 0.0}
            for f in scan_state.no_face_images
        ]
        persons.append(
            {
                "id": -1,
                "label": "No faces",
                "photo_count": len(scan_state.no_face_images),
                "matched": False,
                "ref_similarity": 0.0,
                "images": no_face_imgs,
            }
        )

    return {"has_refs": has_refs, "persons": persons}


@app.get("/api/images/{filename:path}")
def serve_image(filename: str):
    if not scan_state.scan_folder:
        raise HTTPException(status_code=400, detail="No scan has been run yet")

    full_path = Path(scan_state.scan_folder) / filename
    if full_path.is_file():
        return FileResponse(str(full_path))

    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


@app.post("/api/apply")
def apply_selection(req: ApplyRequest):
    if scan_state.status != "done":
        raise HTTPException(status_code=400, detail="Scan is not complete")

    if not req.selected:
        raise HTTPException(status_code=400, detail="No images selected")

    output_path = Path(scan_state.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    errors = []
    for filename in req.selected:
        src = Path(scan_state.scan_folder) / filename
        if not src.is_file():
            errors.append(f"Not found: {filename}")
            continue

        dest = output_path / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(src), str(dest))
            copied += 1
        except Exception as e:
            errors.append(f"{filename}: {e}")

    return {
        "message": f"Copied {copied} image(s) to {scan_state.output_folder}",
        "copied": copied,
        "errors": errors,
    }
