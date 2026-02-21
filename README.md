# Face Cluster

A local web app that scans a folder of photos, groups them by person using face recognition, and lets you pick which ones to keep.

Built for parents who get hundreds of photos dumped into a school/kindergarten WhatsApp group and need to find their kid.

## How it works

1. Point it at a folder of photos
2. Optionally add a "References" folder with photos of specific people you care about
3. It detects faces and clusters them by person
4. If references are provided, matching clusters are shown first — the rest are collapsed under "Other people"
5. Browse person-by-person, select the photos you want
6. Copy selected photos to an output folder

Group shots are deduplicated — each photo appears under the single best-matching person.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app:app
```

Open http://localhost:8000

## Features

- **Face clustering** — groups photos by person automatically using ArcFace embeddings
- **Scan caching** — saves results so you don't rescan the same folder twice
- **Reference filtering** — optionally provide reference photos to only show clusters matching specific people
- **Folder browser** — navigate your filesystem from the UI, no need to type paths
- **Subdirectory support** — recursively scans all subfolders
- **Non-destructive** — copies files to output, never modifies originals

## Requirements

- Python 3.10+
- ~1.5 GB disk space for the face recognition model (downloaded automatically on first run)

## Data pipeline

```mermaid
flowchart TD
    subgraph scan ["Scan (cached)"]
        A[Photos folder] --> B[Face detection]
        B --> C[ArcFace embeddings]
        C --> D[Greedy cosine clustering]
        D --> E[Clusters + centroids]
        E --> F[(JSON cache)]
    end

    subgraph refs ["Reference matching (recomputed)"]
        G[Refs folder] --> H[Face detection]
        H --> I[Ref embeddings]
        I --> J[Score vs centroids]
        E --> J
        J --> K[matched / unmatched]
    end

    subgraph display ["Display"]
        K --> L[Dedup per photo]
        L --> M[Matched first, others collapsed]
    end
```

## How clustering works

Uses [InsightFace](https://github.com/deepinsight/insightface) with the `buffalo_l` model (ArcFace) for face detection and embedding extraction. Faces are grouped using greedy cosine-similarity clustering with a 0.4 threshold. The centroid of each cluster is updated as a running average as new faces are added.

## License

MIT
