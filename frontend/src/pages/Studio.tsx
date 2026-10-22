import { Download, ImageUp, Play, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { postForm, url, type DetectResponse, type Variant } from '../api'
import CompareSlider from '../components/CompareSlider'
import DetectionOverlay, { classColor } from '../components/DetectionOverlay'
import { Alert, Loading, PageHead, Reveal } from '../components/ui'
import { useApi } from '../hooks'

const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']
const SAMPLES = [
  { label: 'Fog', id: 'Fog/foggy-001' }, { label: 'Rain', id: 'Rain/rain_storm-001' },
  { label: 'Snow', id: 'Snow/snow_storm-001' }, { label: 'Sand', id: 'Sand/sand_storm-151' },
]

export function useImageFile() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  useEffect(() => {
    if (!file) { setPreview(null); return }
    const u = URL.createObjectURL(file)
    setPreview(u)
    return () => URL.revokeObjectURL(u)
  }, [file])
  const accept = (f: File | undefined | null) => {
    if (!f) return
    if (!ACCEPT.includes(f.type) && !/\.(jpe?g|png|webp)$/i.test(f.name)) {
      setFileError(`Unsupported file type "${f.type || f.name}". Use JPG, PNG or WebP.`)
      return
    }
    if (f.size > 15 * 2 ** 20) { setFileError('File larger than 15 MB.'); return }
    setFileError(null)
    setFile(f)
  }
  const loadSample = async (id: string) => {
    setFileError(null)
    try {
      const r = await fetch(url(`/api/datasets/dawn/image?id=${encodeURIComponent(id)}&annotated=false&max_size=1600`))
      if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
      const blob = await r.blob()
      setFile(new File([blob], `${id.replace('/', '_')}.jpg`, { type: 'image/jpeg' }))
    } catch (e) {
      setFileError(`Could not load DAWN sample: ${(e as Error).message}`)
    }
  }
  return { file, preview, fileError, accept, loadSample, setFile }
}

export function Dropzone({ onFile, preview, name }: { onFile: (f: File | undefined) => void; preview: string | null; name?: string }) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className={`dropzone ${over ? 'over' : ''}`} role="button" tabIndex={0} aria-label="Upload an image"
      onClick={() => input.current?.click()}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') input.current?.click() }}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files?.[0]) }}>
      <input ref={input} type="file" accept=".jpg,.jpeg,.png,.webp" hidden onChange={(e) => onFile(e.target.files?.[0])} />
      {preview ? (
        <div className="stack" style={{ alignItems: 'center' }}>
          <img src={preview} alt="Selected" style={{ maxHeight: 180, borderRadius: 8 }} />
          <span className="small muted">{name} · click or drop to replace</span>
        </div>
      ) : (
        <div className="stack" style={{ alignItems: 'center' }}>
          <ImageUp size={30} color="var(--cyan)" />
          <b>Drop an image here or click to browse</b>
          <span className="small muted">JPG, PNG or WebP · up to 15 MB</span>
        </div>
      )}
    </div>
  )
}

export function VariantPicker({ variants, variant, setVariant, checkpoint, setCheckpoint, allowB0 = true }: {
  variants: Variant[]; variant: string; setVariant: (v: string) => void; checkpoint: string; setCheckpoint: (c: string) => void; allowB0?: boolean
}) {
  const v = variants.find((x) => x.key === variant)
  return (
    <>
      <label className="field">Processing mode (variant)
        <select value={variant} onChange={(e) => { setVariant(e.target.value); setCheckpoint('') }}>
          {variants.filter((x) => allowB0 || x.key !== 'b0_no_filter').map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
        </select>
      </label>
      <label className="field">Front-end weights
        <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} disabled={variant === 'b0_no_filter'}>
          <option value="">{variant === 'b0_no_filter' ? 'none (no front end)' : 'Untrained (identity-initialised)'}</option>
          {variant !== 'b0_no_filter' && v?.checkpoints.map((c) => <option key={c.path} value={c.path}>{c.run} · seed {c.seed}{c.best_val_map50 != null ? ` · val mAP50 ${(c.best_val_map50 * 100).toFixed(1)}` : ''}</option>)}
        </select>
      </label>
    </>
  )
}

export default function Studio() {
  const variants = useApi<Variant[]>('/api/variants')
  const img = useImageFile()
  const [variant, setVariant] = useState('b0_no_filter')
  const [checkpoint, setCheckpoint] = useState('')
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.6)
  const [classFilter, setClassFilter] = useState('all')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<DetectResponse | null>(null)
  const [showBoxes, setShowBoxes] = useState(true)
  const ctl = useRef<AbortController | null>(null)

  // Prefer a trained checkpoint by default when one exists for the selected variant.
  useEffect(() => {
    const v = variants.data?.find((x) => x.key === variant)
    if (variant !== 'b0_no_filter' && v?.checkpoints.length && !checkpoint) setCheckpoint(v.checkpoints[0].path)
  }, [variant, variants.data]) // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    if (!img.file) return
    setBusy(true); setError(null)
    ctl.current?.abort()
    ctl.current = new AbortController()
    const fd = new FormData()
    fd.append('file', img.file)
    fd.append('variant', variant)
    if (checkpoint) fd.append('checkpoint', checkpoint)
    fd.append('conf', String(conf)); fd.append('iou', String(iou)); fd.append('class_filter', classFilter)
    try {
      setRes(await postForm<DetectResponse>('/api/detect', fd, ctl.current.signal))
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const counts = useMemo(() => {
    const m = new Map<string, { n: number; id: number }>()
    res?.detections.forEach((d) => m.set(d.class_name, { n: (m.get(d.class_name)?.n ?? 0) + 1, id: d.class_id }))
    return [...m.entries()].sort((a, b) => b[1].n - a[1].n)
  }, [res])

  return (
    <>
      <PageHead eyebrow="Live Detection Studio" title="Run the real detector on your image">
        <p>Upload a fog, rain, snow, sandstorm or low-light image. The backend letterboxes it to 448×448, applies the selected
          SG-ERUP front end, runs YOLOv3u and returns boxes in original-image pixels. Nothing here is simulated.</p>
      </PageHead>
      <div className="split">
        <Reveal>
          <div className="card">
            {!res && !busy && (img.preview ? <img src={img.preview} alt="Preview" style={{ borderRadius: 10, margin: '0 auto', maxHeight: 520 }} /> :
              <div className="muted small" style={{ padding: 60, textAlign: 'center' }}>Results appear here. Pick a DAWN sample or upload an image →</div>)}
            {busy && <div className="stack" style={{ padding: 40 }}><Loading label="Running front end + YOLOv3u…" /><div className="progress indeterminate"><div /></div></div>}
            {res && !busy && img.preview && (
              <div className="stack">
                <CompareSlider before={res.files['original.jpg'] ? url(res.files['original.jpg']) : img.preview}
                  after={url(res.files['processed.png'])} beforeLabel="Original" afterLabel={res.frontend.variant === 'b0_no_filter' ? 'Detector input (no front end)' : 'Processed (Î)'}
                  overlay={<DetectionOverlay detections={res.detections} width={res.image.width} height={res.image.height} hidden={!showBoxes} />} />
                <div className="row between">
                  <label className="row small"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Show boxes</label>
                  <div className="row">
                    <a className="btn sm" href={url(res.files['annotated.jpg'])} download><Download size={14} /> Annotated</a>
                    <a className="btn sm" href={url(res.files['processed.png'])} download><Download size={14} /> Processed</a>
                    <a className="btn sm" href={url(res.files['detections.json'])} download><Download size={14} /> JSON</a>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Reveal>

        <Reveal delay={80}>
          <div className="card stack">
            <Dropzone onFile={img.accept} preview={img.preview} name={img.file?.name} />
            <div className="row small"><span className="muted">DAWN samples:</span>
              {SAMPLES.map((s) => <button key={s.id} className="btn sm" onClick={() => img.loadSample(s.id)}>{s.label}</button>)}
            </div>
            {img.fileError && <Alert kind="error">{img.fileError}</Alert>}
            {variants.error && <Alert kind="error">{variants.error}</Alert>}
            {variants.data && <VariantPicker variants={variants.data} variant={variant} setVariant={setVariant} checkpoint={checkpoint} setCheckpoint={setCheckpoint} />}
            <label className="field">Confidence threshold: <b>{conf.toFixed(2)}</b>
              <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
            </label>
            <label className="field">NMS IoU: <b>{iou.toFixed(2)}</b>
              <input type="range" min={0.3} max={0.9} step={0.05} value={iou} onChange={(e) => setIou(Number(e.target.value))} />
            </label>
            <label className="field">Classes
              <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
                <option value="all">All 80 COCO classes</option>
                <option value="protocol5">Paper's 5 classes (person, bicycle, car, bus, motorbike)</option>
              </select>
            </label>
            <div className="row">
              <button className="btn primary" onClick={run} disabled={!img.file || busy}><Play size={16} /> Run Detection</button>
              {res && <button className="btn ghost" onClick={() => { setRes(null); img.setFile(null) }}><RotateCcw size={15} /> Reset</button>}
            </div>
            {error && <Alert kind="error">{error}</Alert>}
          </div>
        </Reveal>
      </div>

      {res && (
        <Reveal className="section">
          <div className="grid g3">
            <div className="card">
              <h3>Detections ({res.detections.length})</h3>
              <div className="row" style={{ marginBottom: 8 }}>{counts.map(([n, c]) => <span key={n} className="pill"><span className="swatch" style={{ background: classColor(c.id) }} />{n} × {c.n}</span>)}</div>
              <div className="det-list">
                {res.detections.length === 0 && <p className="small">No objects above the threshold.</p>}
                {res.detections.map((d, i) => (
                  <div className="det-item" key={i}>
                    <span className="swatch" style={{ background: classColor(d.class_id) }} />
                    <span style={{ width: 92 }}>{d.class_name}</span>
                    <div className="bar-mini"><div style={{ width: `${d.score * 100}%`, background: classColor(d.class_id) }} /></div>
                    <span className="mono" style={{ width: 48, textAlign: 'right' }}>{(d.score * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <h3>Front end</h3>
              <div className="stack small">
                <div><span className="muted">Variant:</span> {variants.data?.find((v) => v.key === res.frontend.variant)?.label}</div>
                <div>{res.frontend.variant === 'b0_no_filter' ? <span className="pill">No processing stage</span> :
                  res.frontend.trained ? <span className="pill good">Trained checkpoint · epoch {res.frontend.epoch} · seed {res.frontend.seed}</span> :
                    <span className="pill warn">Untrained (identity-initialised) — baseline detector with SG-ERUP processing prototype</span>}</div>
                <div><span className="muted">Front-end params:</span> {res.frontend.params.toLocaleString()}</div>
                <div><span className="muted">NSR (Ĩ):</span> {(res.mechanism.nsr * 100).toFixed(3)}% · <span className="muted">BRB (Ĩ):</span> {res.mechanism.brb.toFixed(4)}</div>
                <div><span className="muted">Device:</span> {res.device} · input {res.imgsz}×{res.imgsz}</div>
              </div>
            </div>
            <div className="card">
              <h3>Timing (this request)</h3>
              <table className="tbl"><tbody>
                {Object.entries(res.timing_ms).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, ' ')}</td><td className="num">{v.toFixed(1)} ms</td></tr>)}
              </tbody></table>
              <p className="tiny muted" style={{ marginTop: 8 }}>Single request wall time including Python overhead; see Model & Experiments for the E7 benchmark.</p>
            </div>
          </div>
        </Reveal>
      )}
    </>
  )
}
