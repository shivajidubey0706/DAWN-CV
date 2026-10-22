import { Cpu, Play, Rocket } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { sendJSON, url, type ModelStatus, type RunInfo, type Variant } from '../api'
import { Lines } from '../components/charts'
import { Alert, Kpi, Loading, PageHead, Reveal, fmtNum } from '../components/ui'
import { useApi } from '../hooks'

interface RunDetail {
  status: Record<string, unknown> | null
  history: { epochs: { epoch: number; train: Record<string, number>; val: { loss: number; map50: number | null; map50_95: number | null; n_images: number }; lr: number; time_s: number }[] } | null
  config: Record<string, Record<string, unknown> | string> | null
  environment: Record<string, unknown> | null
  steps: { epoch: number; step: number; loss: number; l_det: number; l_sat: number; l_tv: number; gate_mean: number; lr: number }[]
  error: string | null
  has_loss_curve: boolean
  split: Record<string, unknown>
}
interface Jobs { enabled: boolean; configs: string[]; jobs: { id: string; config: string; variant: string; seed: number; state: string; log_tail: string; cmd: string; run_status: Record<string, unknown> | null }[] }
interface Checks { bpw_slope_both_cps_at_0_1_L8?: number; encoder_params?: number; encoder_gmac?: Record<string, number>; gate_params?: number; bpw_slope_random_params?: { median: number; p90: number }; status?: string }

function stateClass(s?: string) { return s === 'done' ? 'good' : s === 'failed' ? 'bad' : s === 'training' ? 'info' : '' }

export default function Experiments() {
  const model = useApi<ModelStatus>('/api/model/status', 20000)
  const sys = useApi<Record<string, unknown>>('/api/system')
  const runs = useApi<RunInfo[]>('/api/training/runs', 8000)
  const checks = useApi<Checks>('/api/checks')
  const variants = useApi<Variant[]>('/api/variants')
  const jobs = useApi<Jobs>('/api/training/jobs', 5000)
  const [sel, setSel] = useState<string>('')
  const detail = useApi<RunDetail>(sel ? `/api/training/runs/${sel}` : null, 10000)
  const [jobVariant, setJobVariant] = useState('sg_full')
  const [jobMsg, setJobMsg] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)

  useEffect(() => {
    if (!sel && runs.data?.length) {
      const r = runs.data.find((x) => x.run === 'short_real' && x.variant === 'sg_full') ?? runs.data.find((x) => x.status.state === 'training') ?? runs.data[0]
      setSel(`${r.run}/${r.variant}/${r.seed}`)
    }
  }, [runs.data, sel])

  const epochData = useMemo(() => (detail.data?.history?.epochs ?? []).map((e) => ({
    epoch: e.epoch + 1, 'train L_total': +e.train.loss.toFixed(4), 'train L_det': +e.train.l_det.toFixed(4), 'val L_det': +e.val.loss.toFixed(4),
    'val mAP@0.5 (%)': e.val.map50 != null ? +(e.val.map50 * 100).toFixed(2) : null,
  })), [detail.data])
  const stepData = useMemo(() => (detail.data?.steps ?? []).map((s, i) => ({ i, loss: +s.loss.toFixed(4), 'gate mean': +s.gate_mean.toFixed(4), 'L_sat×100': +(s.l_sat * 100).toFixed(5) })), [detail.data])

  async function startJob() {
    setJobMsg(null)
    try {
      const j = await sendJSON<{ id: string; cmd: string }>('/api/training/jobs', { config: 'smoke', variant: jobVariant, seed: 0 })
      setJobMsg({ kind: 'info', text: `Started job ${j.id}: ${j.cmd}` }); jobs.reload()
    } catch (e) { setJobMsg({ kind: 'error', text: (e as Error).message }) }
  }

  const det = model.data?.detector
  const cfg = detail.data?.config
  return (
    <>
      <PageHead eyebrow="Model & Experiments" title="Architecture, checkpoints, training runs and hardware">
        <p>{model.data?.status_note}</p>
      </PageHead>
      {model.error && <Alert kind="error">{model.error}</Alert>}

      <div className="grid g3">
        <Reveal><div className="card" style={{ height: '100%' }}>
          <h3>Detector</h3>
          {model.data?.detector_error && <Alert kind="error">{model.data.detector_error}</Alert>}
          {det ? (
            <div className="stack small">
              <span className="pill good">loaded on {det.device}</span>
              <div>{det.architecture}</div>
              <div className="grid g2"><Kpi value={`${(det.params / 1e6).toFixed(2)} M`} label="parameters" /><Kpi value={det.imgsz} label="input size" /></div>
              <div className="muted tiny">weights: {det.weights} ({model.data?.weights_mb} MB) · {det.n_classes} COCO classes</div>
            </div>
          ) : model.data?.detector_loading ? <Loading label="Loading detector…" /> : <p className="small">{model.data?.weights_exist ? 'Not loaded yet.' : 'Weights missing — run python scripts/setup_weights.py'}</p>}
        </div></Reveal>
        <Reveal delay={60}><div className="card" style={{ height: '100%' }}>
          <h3>SG-ERUP front end</h3>
          <div className="stack small">
            <div className="grid g2">
              <Kpi value={checks.data?.encoder_params ? fmtNum(checks.data.encoder_params) : '6,781,042'} label="encoder params" />
              <Kpi value={checks.data?.gate_params ?? 1028} label="gate params" />
            </div>
            {checks.data?.encoder_gmac && <div>Encoder MACs: {Object.entries(checks.data.encoder_gmac).map(([k, v]) => `${v.toFixed(2)} G @ ${k}`).join(' · ')} <span className="muted">(paper: 4.95 / 15.14)</span></div>}
            {checks.data?.bpw_slope_both_cps_at_0_1_L8 && <div>BPW max slope (cps at (0,1), L=8): <b>{checks.data.bpw_slope_both_cps_at_0_1_L8.toFixed(1)}</b> <span className="muted">(paper ≈ 169)</span></div>}
            {checks.data?.bpw_slope_random_params && <div>Random-parameter slope median {checks.data.bpw_slope_random_params.median.toFixed(1)} <span className="muted">(paper ≈ 50 — not reproduced, see Docs A1)</span></div>}
            <div className="muted tiny">Encoder: 5×(3×3 conv + LeakyReLU + maxpool), 64→1024 ch, GAP, FC→498 (12 BPW + 486 KBL)</div>
          </div>
        </div></Reveal>
        <Reveal delay={120}><div className="card" style={{ height: '100%' }}>
          <h3><Cpu size={17} /> Hardware</h3>
          {sys.data ? (
            <div className="stack small">
              <div><b>{String(sys.data.gpu ?? 'No CUDA GPU')}</b>{sys.data.gpu_memory_total_gb ? ` · ${sys.data.gpu_memory_free_gb} / ${sys.data.gpu_memory_total_gb} GB free` : ''}</div>
              <div className="muted">CUDA {String(sys.data.cuda)} · cuDNN {String(sys.data.cudnn)} · torch {String(sys.data.torch)}</div>
              <div className="muted">ultralytics {String(sys.data.ultralytics)} · Python {String(sys.data.python)}</div>
              <div className="muted tiny">{String(sys.data.platform)}</div>
            </div>
          ) : <Loading />}
        </div></Reveal>
      </div>

      <Reveal className="section">
        <div className="card">
          <h3>Checkpoints</h3>
          <div className="table-wrap" style={{ maxHeight: 340 }}>
            <table className="tbl"><thead><tr><th>Run</th><th>Variant</th><th className="num">Seed</th><th>State</th><th className="num">Best val mAP@0.5</th><th className="num">Size</th><th>Path</th></tr></thead>
              <tbody>{(model.data?.checkpoints ?? []).map((c) => (
                <tr key={c.path}><td>{c.run}</td><td>{variants.data?.find((v) => v.key === c.variant)?.label ?? c.variant}</td><td className="num">{c.seed}</td>
                  <td><span className={`pill ${stateClass(c.state)}`}>{c.state}</span></td>
                  <td className="num">{c.best_val_map50 != null ? (c.best_val_map50 * 100).toFixed(2) : '—'}</td><td className="num">{c.size_mb} MB</td><td className="mono tiny">{c.path}</td></tr>))}
              </tbody></table>
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>Best checkpoint is selected on the synthetic validation split (10% of VOC trainval with fog / gamma / SHA), never on test data.</p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card">
          <h3>Training runs</h3>
          {runs.error && <Alert kind="error">{runs.error}</Alert>}
          <div className="table-wrap" style={{ maxHeight: 360 }}>
            <table className="tbl"><thead><tr><th>Run</th><th>Variant</th><th>Seed</th><th>State</th><th>Progress</th><th className="num">Epochs</th><th className="num">λ_s</th><th className="num">p_sha</th><th></th></tr></thead>
              <tbody>{(runs.data ?? []).map((r) => {
                const key = `${r.run}/${r.variant}/${r.seed}`
                const p = (r.status.progress as number | undefined) ?? 0
                return (
                  <tr key={key} style={{ background: key === sel ? 'rgba(34,211,238,.06)' : undefined }}>
                    <td>{r.run}</td><td>{r.variant}</td><td>{r.seed}</td>
                    <td><span className={`pill ${stateClass(r.status.state)}`}>{r.status.state}</span>{r.has_error && <span className="pill bad" style={{ marginLeft: 4 }}>error</span>}</td>
                    <td style={{ minWidth: 120 }}><div className="progress"><div style={{ width: `${p * 100}%` }} /></div><span className="tiny muted">{(p * 100).toFixed(0)}%{r.status.eta_s && r.status.state === 'training' ? ` · ETA ${Math.round(Number(r.status.eta_s) / 60)} min` : ''}</span></td>
                    <td className="num">{r.epochs_done}/{String(r.train?.epochs ?? '—')}</td><td className="num">{String(r.loss?.lambda_s ?? '—')}</td><td className="num">{String(r.aug?.p_sha ?? '—')}</td>
                    <td><button className="btn sm ghost" onClick={() => setSel(key)}>View</button></td>
                  </tr>)
              })}</tbody></table>
          </div>
        </div>
      </Reveal>

      {sel && (
        <Reveal className="section">
          <div className="card stack">
            <h3>Run {sel}</h3>
            {detail.loading && !detail.data && <Loading />}
            {detail.data?.error && <Alert kind="error"><b>Error log</b><pre className="mono tiny" style={{ whiteSpace: 'pre-wrap' }}>{detail.data.error}</pre></Alert>}
            {detail.data?.status?.note != null && <Alert kind="info">{String(detail.data.status.note)}</Alert>}
            <div className="grid g2">
              <div><div className="small muted">Per-epoch losses (training set vs fixed synthetic validation split)</div>
                <Lines data={epochData} x="epoch" series={[
                  { key: 'train L_total', name: 'train L_total', color: '#3987e5' }, { key: 'train L_det', name: 'train L_det', color: '#199e70', dashed: true },
                  { key: 'val L_det', name: 'val L_det', color: '#d95926' }]} yLabel="loss" /></div>
              <div><div className="small muted">Validation mAP@0.5 (synthetic val split, checkpoint selection)</div>
                <Lines data={epochData} x="epoch" series={[{ key: 'val mAP@0.5 (%)', name: 'val mAP@0.5 (%)', color: '#9085e9' }]} yLabel="%" /></div>
              <div><div className="small muted">Step loss (every {String((cfg?.train as Record<string, unknown> | undefined)?.log_every ?? 20)} steps)</div>
                <Lines data={stepData} x="i" series={[{ key: 'loss', name: 'loss', color: '#3987e5' }]} /></div>
              <div><div className="small muted">Mean gate value during training</div>
                <Lines data={stepData} x="i" series={[{ key: 'gate mean', name: 'gate mean', color: '#c98500' }]} /></div>
            </div>
            {detail.data?.has_loss_curve && <details><summary className="small">Saved loss_curve.png</summary><img src={url(`/api/training/runs/${sel}/loss_curve.png`)} alt="Loss curve" style={{ maxWidth: 760 }} /></details>}
            <div className="grid g2">
              <details><summary className="small">Training configuration</summary><pre className="mono tiny" style={{ maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(cfg, null, 2)}</pre></details>
              <details><summary className="small">Environment record & split</summary><pre className="mono tiny" style={{ maxHeight: 300, overflow: 'auto' }}>{JSON.stringify({ environment: detail.data?.environment, split: detail.data?.split }, null, 2)}</pre></details>
            </div>
          </div>
        </Reveal>
      )}

      <Reveal>
        <div className="card stack">
          <h3><Rocket size={17} /> Launch a smoke training job</h3>
          <p className="small" style={{ margin: 0 }}>Runs <code>scripts/train.py --config configs/smoke.yaml</code> (64 images, 2 epochs) as a background process. Results of smoke runs are pipeline checks, not research results. The GPU is shared with any running job.</p>
          {jobs.data && !jobs.data.enabled && <Alert kind="warn">Training jobs are disabled on this server.</Alert>}
          <div className="row">
            <select value={jobVariant} onChange={(e) => setJobVariant(e.target.value)} style={{ width: 'auto' }} aria-label="Variant">
              {(variants.data ?? []).map((v) => <option key={v.key} value={v.key}>{v.label}</option>)}
            </select>
            <button className="btn primary" onClick={startJob} disabled={!jobs.data?.enabled || jobs.data?.jobs.some((j) => j.state === 'running')}><Play size={15} /> Start</button>
          </div>
          {jobMsg && <Alert kind={jobMsg.kind}>{jobMsg.text}</Alert>}
          {jobs.data?.jobs.map((j) => (
            <div key={j.id} className="card" style={{ background: 'var(--bg-2)' }}>
              <div className="row between"><b>Job {j.id} · {j.variant}</b><span className={`pill ${j.state === 'succeeded' ? 'good' : j.state === 'failed' ? 'bad' : 'info'}`}>{j.state}</span></div>
              {j.state === 'running' && <div className="progress" style={{ margin: '8px 0' }}><div style={{ width: `${Number(j.run_status?.progress ?? 0) * 100}%` }} /></div>}
              <pre className="mono tiny" style={{ maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{j.log_tail.split('\n').filter((l) => !/warn/i.test(l)).slice(-14).join('\n')}</pre>
            </div>
          ))}
        </div>
      </Reveal>
    </>
  )
}
