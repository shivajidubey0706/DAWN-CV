// Typed client for the FastAPI backend. All requests go through the same origin (/api),
// proxied by Vite in development; no API keys ever live in the frontend.

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, { signal })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(0, 'Cannot reach the backend. Is it running on port 8000? (see README)')
  }
  return handle<T>(res)
}

export async function sendJSON<T>(path: string, body: unknown, method = 'POST'): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the backend. Is it running on port 8000?')
  }
  return handle<T>(res)
}

export async function postForm<T>(path: string, form: FormData, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, { method: 'POST', body: form, signal })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(0, 'Cannot reach the backend. Is it running on port 8000?')
  }
  return handle<T>(res)
}

export const url = (p: string) => API_BASE + p

// ---------------------------------------------------------------- types
export interface Detection {
  box: [number, number, number, number]
  score: number
  class_id: number
  class_name: string
}

export interface Mechanism {
  nsr: number
  brb: number
  nsr_final: number
  brb_final: number
  bright_weight_mean: number
}

export interface FrontendInfo {
  variant: string
  checkpoint: string | null
  trained: boolean
  params: number
  epoch?: number
  seed?: number
}

export interface DetectResponse {
  id: string
  detections: Detection[]
  timing_ms: Record<string, number>
  frontend: FrontendInfo
  mechanism: Mechanism
  device: string
  imgsz: number
  image: { width: number; height: number; filename: string }
  files: Record<string, string>
  settings: Record<string, unknown>
}

export interface ProcessResponse {
  maps: Record<string, string>
  mechanism: Mechanism
  detections: Detection[]
  timing_ms: Record<string, number>
  gate_stats: { mean: number; min: number; max: number; lowres: number[][] }
  theta_b: number[] | null
  provenance: string
  gate_note: string
  gate_mode: string
  frontend: FrontendInfo
  settings: Record<string, string | number>
}

export interface Checkpoint {
  path: string
  run: string
  variant: string
  seed: number
  state: string
  best_val_map50: number | null
  size_mb: number
}

export interface Variant {
  key: string
  label: string
  paper_id: string
  description?: string
  frontend: Record<string, unknown>
  loss: Record<string, number | string>
  aug: Record<string, number>
  checkpoints: Checkpoint[]
}

export interface ModelStatus {
  detector_loaded: boolean
  detector_loading: boolean
  detector_error: string | null
  weights_path: string
  weights_exist: boolean
  weights_mb: number | null
  detector: null | { weights: string; params: number; n_classes: number; device: string; imgsz: number; architecture: string }
  checkpoints: Checkpoint[]
  status_note: string
}

export interface Stat {
  n: number
  mean: number | null
  std: number | null
  ci: [number, number] | null
}

export interface SummaryRow {
  variant: string
  label: string
  dataset: string
  subset: string
  n_seeds: number
  seeds: number[]
  n_images: number
  n_gt: number
  trained: boolean
  [metric: string]: Stat | string | number | number[] | boolean
}

export interface Comparison {
  variant: string
  reference: string
  dataset: string
  subset: string
  n_pairs: number
  diff: Stat
  wilcoxon_p: number | null
  note: string | null
}

export interface ExperimentSummary {
  experiment: string
  n_result_files: number
  reference: string
  rows: SummaryRow[]
  comparisons: Comparison[]
  variants_evaluated: string[]
  datasets: string[]
  seeds: number[]
  generated_at: string
  config_file?: string
}

export interface ExperimentFile {
  file: string
  variant: string
  seed: number
  dataset: string
  trained: boolean
  checkpoint: string | null
  n_images: number
  eval_seconds: number
  settings: Record<string, unknown>
  per_class: Record<string, { ap50: number | null; ap_coco: number | null; n_gt: number }>
  sizes: Record<string, { ap: number | null; ap50: number | null; n_gt: number }>
  bootstrap?: { n: number; ci95: [number, number]; std: number }
  evaluated_at: string
  environment: Record<string, string>
}

export interface DatasetReport {
  name: string
  status: string
  available: boolean
  n_images_found?: number
  n_valid_samples?: number
  n_missing_annotations?: number
  n_corrupt_images?: number | null
  n_protocol_boxes?: number
  n_ignore_boxes?: number
  n_images_without_protocol_objects?: number
  raw_class_counts?: Record<string, number>
  protocol_class_counts?: Record<string, number>
  weather_counts?: Record<string, number>
  image_size?: { min: number[]; max: number[]; median: number[] }
  n_issues?: number
  issues?: string[]
  scanned_at?: string
  message?: string
  scan_seconds?: number
}

export interface DatasetEntry {
  name: string
  title: string
  format: string
  root: string
  available: boolean
  role: string
  url: string
  notes: string
  class_set: string
  reports: Record<string, DatasetReport | null>
}

export interface SampleItem {
  id: string
  weather: string | null
  width: number
  height: number
  n_objects: number
  n_ignore: number
  labels: string[]
}

export interface RunInfo {
  run: string
  variant: string
  seed: string
  status: Record<string, unknown> & { state?: string; progress?: number; error?: string }
  epochs_done: number
  has_error: boolean
  train?: Record<string, unknown>
  loss?: Record<string, unknown>
  aug?: Record<string, unknown>
}
