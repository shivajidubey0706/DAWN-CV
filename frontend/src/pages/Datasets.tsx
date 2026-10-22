import { CheckCircle2, FolderCog, RefreshCw, ShieldCheck, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { getJSON, sendJSON, url, type DatasetEntry, type DatasetReport, type SampleItem } from '../api'
import { HBars } from '../components/charts'
import { Alert, Kpi, Loading, PageHead, Reveal, fmtNum } from '../components/ui'
import { useApi } from '../hooks'

interface DatasetsResp { datasets: DatasetEntry[]; leakage_check: { disjoint: boolean; overlaps: Record<string, number>; sizes: Record<string, number> } | null }
interface SamplesResp { total: number; offset: number; split: string | null; items: SampleItem[] }

function ReportView({ rep }: { rep: DatasetReport }) {
  if (!rep.available) return <Alert kind="warn">{rep.message}</Alert>
  const cls = Object.entries(rep.protocol_class_counts ?? {}).sort((a, b) => b[1] - a[1]).map(([name, value]) => ({ name, value }))
  const raw = Object.entries(rep.raw_class_counts ?? {}).map(([name, value]) => ({ name, value }))
  const wx = Object.entries(rep.weather_counts ?? {}).map(([name, value]) => ({ name, value }))
  return (
    <div className="stack">
      <div className="grid g4">
        <Kpi value={fmtNum(rep.n_valid_samples)} label="valid images" />
        <Kpi value={fmtNum(rep.n_protocol_boxes)} label="protocol boxes" />
        <Kpi value={fmtNum(rep.n_ignore_boxes)} label="ignore regions" />
        <Kpi value={fmtNum(rep.n_missing_annotations)} label="missing annotations" />
      </div>
      <div className="row small">
        <span className={`pill ${rep.status === 'ok' ? 'good' : 'warn'}`}>{rep.status}</span>
        <span className="pill">corrupt images: {rep.n_corrupt_images ?? 'not checked'}</span>
        <span className="pill">without protocol objects: {fmtNum(rep.n_images_without_protocol_objects)}</span>
        {rep.image_size && <span className="pill">median size {rep.image_size.median.join('×')}</span>}
        <span className="pill">scanned {rep.scanned_at} ({rep.scan_seconds}s)</span>
      </div>
      <div className="grid g3">
        <div><div className="small muted">Protocol class distribution</div><HBars data={cls} /></div>
        <div><div className="small muted">Raw labels in annotation files</div><HBars data={raw} color="#9085e9" /></div>
        <div><div className="small muted">Images per weather subset</div><HBars data={wx} color="#199e70" /></div>
      </div>
      {!!rep.n_issues && (
        <details><summary className="small">{rep.n_issues} validation issue(s)</summary>
          <pre className="mono tiny" style={{ maxHeight: 200, overflow: 'auto' }}>{rep.issues?.join('\n')}</pre></details>
      )}
    </div>
  )
}

export default function Datasets() {
  const ds = useApi<DatasetsResp>('/api/datasets')
  const [sel, setSel] = useState('dawn')
  const [busy, setBusy] = useState<string | null>(null)
  const [msg, setMsg] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)
  const [path, setPath] = useState('')
  const [verify, setVerify] = useState(false)
  const [weather, setWeather] = useState('')
  const [cls, setCls] = useState('')
  const [split, setSplit] = useState('')
  const [offset, setOffset] = useState(0)
  const [samples, setSamples] = useState<SamplesResp | null>(null)
  const [sampleErr, setSampleErr] = useState<string | null>(null)
  const [open, setOpen] = useState<SampleItem | null>(null)
  const [ann, setAnn] = useState<{ boxes: number[][]; labels: string[]; raw_labels: string[] } | null>(null)

  const entry = ds.data?.datasets.find((d) => d.name === sel)
  const isVoc = entry?.format === 'voc'
  const reportKey = isVoc ? (split || 'test') : 'all'
  const rep = entry?.reports[reportKey] ?? null
  useEffect(() => { setPath(entry?.root ?? ''); setOffset(0); setWeather(''); setCls(''); setSplit('') }, [sel, entry?.root])

  useEffect(() => {
    if (!entry?.available) { setSamples(null); return }
    const q = new URLSearchParams({ offset: String(offset), limit: '24' })
    if (weather) q.set('weather', weather)
    if (cls) q.set('cls', cls)
    if (isVoc) q.set('split', split || 'test')
    setSampleErr(null)
    getJSON<SamplesResp>(`/api/datasets/${sel}/samples?${q}`).then(setSamples).catch((e) => setSampleErr(e.message))
  }, [sel, weather, cls, offset, split, entry?.available, isVoc])

  useEffect(() => {
    setAnn(null)
    if (!open) return
    const q = new URLSearchParams({ id: open.id })
    if (isVoc) q.set('split', split || 'test')
    getJSON<typeof ann>(`/api/datasets/${sel}/annotation?${q}`).then(setAnn).catch(() => setAnn(null))
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  async function validate() {
    setBusy('validate'); setMsg(null)
    try {
      const q = new URLSearchParams({ verify_images: String(verify) })
      if (isVoc) q.set('split', split || 'test')
      const r = await sendJSON<DatasetReport>(`/api/datasets/${sel}/validate?${q}`, {})
      setMsg({ kind: 'info', text: `Validation finished: ${r.status}, ${fmtNum(r.n_valid_samples)} images, ${r.n_issues ?? 0} issue(s).` })
      ds.reload()
    } catch (e) { setMsg({ kind: 'error', text: (e as Error).message }) } finally { setBusy(null) }
  }
  async function savePath() {
    setBusy('path'); setMsg(null)
    try {
      const r = await sendJSON<{ message: string }>(`/api/datasets/${sel}/path`, { root: path }, 'PUT')
      setMsg({ kind: 'info', text: r.message }); ds.reload()
    } catch (e) { setMsg({ kind: 'error', text: (e as Error).message }) } finally { setBusy(null) }
  }

  const weathers = Object.keys(rep?.weather_counts ?? {})
  const classes = Object.keys(rep?.protocol_class_counts ?? {})
  const q = (id: string, annotated: boolean, size: number) =>
    url(`/api/datasets/${sel}/image?${new URLSearchParams({ id, annotated: String(annotated), max_size: String(size), ...(isVoc ? { split: split || 'test' } : {}) })}`)

  return (
    <>
      <PageHead eyebrow="Dataset Explorer" title="Datasets scanned from disk">
        <p>Counts, class distributions and samples come from scanning the files under each dataset root. A dataset that is not on disk is
          shown as unavailable with its official download page — nothing is assumed or downloaded silently.</p>
      </PageHead>
      {ds.error && <Alert kind="error">{ds.error}</Alert>}
      {ds.loading && !ds.data && <Loading />}

      <Reveal>
        <div className="row" role="tablist">
          {ds.data?.datasets.map((d) => (
            <button key={d.name} className={`btn sm ${sel === d.name ? '' : 'ghost'}`} onClick={() => setSel(d.name)} role="tab" aria-selected={sel === d.name}>
              <span className={`status-dot ${d.available ? 'ok' : ''}`} /> {d.name}
            </button>
          ))}
        </div>
      </Reveal>

      {ds.data?.leakage_check && (
        <Reveal><div style={{ marginTop: 14 }}>
          <Alert kind={ds.data.leakage_check.disjoint ? 'info' : 'error'}>
            <ShieldCheck size={15} style={{ verticalAlign: -3 }} /> <b>Split leakage check:</b>{' '}
            {ds.data.leakage_check.disjoint ? 'PASS — no image shared between ' : 'FAIL — overlaps between '}
            {Object.entries(ds.data.leakage_check.sizes).map(([k, v]) => `${k} (${fmtNum(v)})`).join(', ')}.
          </Alert>
        </div></Reveal>
      )}

      {entry && (
        <div className="section stack">
          <Reveal>
            <div className="card stack">
              <div className="row between">
                <div><h2 style={{ marginBottom: 2 }}>{entry.title}</h2><div className="small muted">format: {entry.format} · role: {entry.role} · class set: {entry.class_set}</div></div>
                {entry.available ? <span className="pill good"><CheckCircle2 size={13} /> available</span> : <span className="pill warn">not downloaded</span>}
              </div>
              <p className="small" style={{ margin: 0 }}>{entry.notes} {entry.url && <a href={entry.url} target="_blank" rel="noreferrer">Official page ↗</a>}</p>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <label className="field" style={{ flex: 1, minWidth: 260 }}>Dataset root path
                  <input type="text" value={path} onChange={(e) => setPath(e.target.value)} spellCheck={false} />
                </label>
                <button className="btn" onClick={savePath} disabled={busy !== null}><FolderCog size={15} /> Save path</button>
                {isVoc && <label className="field">Split<select value={split || 'test'} onChange={(e) => { setSplit(e.target.value); setOffset(0) }}><option value="trainval">trainval</option><option value="test">test</option></select></label>}
                <label className="row small"><input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} /> decode-check every image</label>
                <button className="btn primary" onClick={validate} disabled={busy !== null || !entry.available}>
                  {busy === 'validate' ? <div className="spinner" /> : <RefreshCw size={15} />} Validate</button>
              </div>
              {msg && <Alert kind={msg.kind}>{msg.text}</Alert>}
            </div>
          </Reveal>

          <Reveal>
            <div className="card">
              <h3>Validation report {isVoc && <span className="pill">{split || 'test'}</span>}</h3>
              {rep ? <ReportView rep={rep} /> : <p className="small muted">{entry.available ? 'No report yet — click Validate.' : 'Dataset not on disk.'}</p>}
            </div>
          </Reveal>

          {entry.available && (
            <Reveal>
              <div className="card">
                <div className="row between" style={{ marginBottom: 12 }}>
                  <h3 style={{ margin: 0 }}>Sample gallery {samples && <span className="muted small">({fmtNum(samples.total)} matching)</span>}</h3>
                  <div className="row">
                    {!!weathers.length && <select value={weather} onChange={(e) => { setWeather(e.target.value); setOffset(0) }} aria-label="Weather filter" style={{ width: 'auto' }}>
                      <option value="">All weather</option>{weathers.map((w) => <option key={w}>{w}</option>)}</select>}
                    <select value={cls} onChange={(e) => { setCls(e.target.value); setOffset(0) }} aria-label="Class filter" style={{ width: 'auto' }}>
                      <option value="">All classes</option>{classes.map((c) => <option key={c}>{c}</option>)}</select>
                  </div>
                </div>
                {sampleErr && <Alert kind="error">{sampleErr}</Alert>}
                <div className="gallery">
                  {samples?.items.map((s) => (
                    <button key={s.id} className="thumb" onClick={() => setOpen(s)}>
                      <img loading="lazy" src={q(s.id, true, 420)} alt={s.id} />
                      <div className="cap">{s.id.split('/').pop()} · {s.weather ?? ''} · {s.n_objects} obj{s.n_ignore ? ` (+${s.n_ignore} ignore)` : ''}</div>
                    </button>
                  ))}
                </div>
                {samples && samples.total > 24 && (
                  <div className="row" style={{ justifyContent: 'center', marginTop: 14 }}>
                    <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 24))}>Previous</button>
                    <span className="small muted">{offset + 1}–{Math.min(offset + 24, samples.total)} of {fmtNum(samples.total)}</span>
                    <button className="btn sm" disabled={offset + 24 >= samples.total} onClick={() => setOffset(offset + 24)}>Next</button>
                  </div>
                )}
              </div>
            </Reveal>
          )}
        </div>
      )}

      {open && (
        <div className="modal-back" onClick={() => setOpen(null)} role="dialog" aria-modal="true" aria-label={open.id}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="row between"><b>{open.id}</b><button className="btn ghost sm" onClick={() => setOpen(null)} aria-label="Close"><X size={16} /></button></div>
            <img src={q(open.id, true, 1400)} alt={open.id} style={{ margin: '12px auto', maxHeight: '64vh' }} />
            <div className="small muted">{open.width}×{open.height} · weather: {open.weather ?? '—'} · cyan = protocol class, grey = ignore region</div>
            {ann && (
              <div className="table-wrap" style={{ marginTop: 10, maxHeight: 220 }}>
                <table className="tbl"><thead><tr><th>#</th><th>Label (mapped)</th><th className="num">x1</th><th className="num">y1</th><th className="num">x2</th><th className="num">y2</th></tr></thead>
                  <tbody>{ann.boxes.map((b, i) => <tr key={i}><td>{i + 1}</td><td>{ann.labels[i]}</td>{b.map((v, j) => <td key={j} className="num">{v.toFixed(0)}</td>)}</tr>)}</tbody></table>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
