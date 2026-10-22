import { motion } from 'framer-motion'
import { ArrowRight, Beaker, Brain, CheckCircle2, CircleDashed, Database, Gauge, Layers, ScanSearch, ShieldAlert, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { DatasetEntry, ModelStatus } from '../api'
import { Alert, Kpi, Reveal, fmtNum } from '../components/ui'
import { useApi } from '../hooks'

interface PaperMeta { title: string; status: string; abstract: string; problem: string; research_questions: string[]; hypotheses: string[]; gaps: { gap: string; status: string; closes: string }[] }
type ExpList = { name: string; n_result_files: number; variants: string[] | null; seeds: number[] | null }[]

const PIPE = [
  { t: 'Input I', s: 'RGB, [0,1]', k: '' },
  { t: 'Encoder E', s: '5 conv → θB, θK', k: 'old' },
  { t: 'Cue maps', s: 'μ, δ, ρ', k: 'new' },
  { t: 'Gate g → G', s: '1×1 conv + σ, bilinear', k: 'new' },
  { t: 'BPW blend', s: 'G·B(I) + (1−G)·I', k: 'new' },
  { t: 'KBL', s: 'local 9×9 filter', k: 'old' },
  { t: 'YOLOv3 D', s: 'boxes', k: 'old' },
]

export default function Home() {
  const paper = useApi<{ meta: PaperMeta }>('/api/paper')
  const model = useApi<ModelStatus>('/api/model/status')
  const ds = useApi<{ datasets: DatasetEntry[] }>('/api/datasets')
  const exps = useApi<ExpList>('/api/experiments')
  const meta = paper.data?.meta
  const dawn = ds.data?.datasets.find((d) => d.name === 'dawn')?.reports.all
  const trained = model.data?.checkpoints.filter((c) => c.state === 'done' && c.variant !== 'b0_no_filter' && c.run !== 'smoke') ?? []
  const realExp = exps.data?.find((e) => e.name === 'short_real')

  const status: { label: string; done: boolean | null; note: string }[] = [
    { label: 'SG-ERUP modules (BPW, KBL, cues, gate, L_sat, L_tv, SHA)', done: true, note: 'Implemented + unit tested; analytic checks reproduce the paper (slope 169, 6.78 M encoder params, 1,028 gate params).' },
    { label: 'Real YOLOv3 detector', done: model.data ? model.data.weights_exist : null, note: model.data?.detector?.architecture ?? 'Ultralytics YOLOv3u, COCO-pretrained (not original Darknet head).' },
    { label: 'DAWN dataset', done: ds.data ? !!dawn?.n_valid_samples : null, note: dawn?.n_valid_samples ? `${fmtNum(dawn.n_valid_samples)} images, ${fmtNum(dawn.n_protocol_boxes)} protocol boxes scanned from disk.` : 'Not found.' },
    { label: 'Trained front-end checkpoints', done: model.data ? trained.length > 0 : null, note: trained.length ? `${trained.length} run(s) from the reduced short_real protocol (frozen detector).` : 'None yet — front ends run identity-initialised.' },
    { label: 'Evaluation results', done: exps.data ? !!realExp?.n_result_files : null, note: realExp ? `${realExp.n_result_files} result file(s) in short_real.` : 'No research results yet.' },
    { label: 'Full paper protocol (8 seeds, 80 epochs, joint training, RTTS/ExDark/BDD/ACDC)', done: false, note: 'Not run — exceeds the 4 GB GPU; configs/paper_full.yaml is provided.' },
  ]

  return (
    <>
      <section className="hero">
        <div>
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .6 }}>
            <div className="eyebrow">Research prototype · image-adaptive detection</div>
            <h1>Spatially gated, <span className="grad">saturation-aware</span> processing for detection in adverse weather</h1>
            <p className="hero-lede">
              SG-ERUP extends ERUP-YOLO with a ~1k-parameter spatial gate that decides <i>where</i> its global
              Bézier tone curve may brighten the image, a one-sided saturation penalty, and spatially heterogeneous
              augmentation — aimed at ERUP-YOLO's documented regression on DAWN sandstorm scenes.
            </p>
            <div className="row" style={{ marginTop: 18 }}>
              <Link className="btn primary" to="/studio"><ScanSearch size={17} /> Run detection</Link>
              <Link className="btn" to="/lab"><Layers size={17} /> Open processing lab</Link>
              <Link className="btn ghost" to="/analytics">Measured results <ArrowRight size={16} /></Link>
            </div>
          </motion.div>
        </div>
        <motion.div className="hero-visual" initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .7, delay: .15 }}>
          <img src="/api/datasets/dawn/image?id=Sand/sand_storm-151&max_size=900" alt="DAWN sandstorm image with its ground-truth boxes"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')} />
          <div className="scanline" />
          <span className="tag pill" style={{ position: 'absolute', left: 12, bottom: 12, background: 'rgba(7,11,22,.8)' }}>
            DAWN · Sand · ground-truth annotations
          </span>
        </motion.div>
      </section>

      <Alert kind="warn">
        <b>Status: hypothesis, not a validated improvement.</b> {meta?.status ?? 'The paper proposes SG-ERUP without experimental results.'}{' '}
        This project implements and tests it; any accuracy number on this site comes from saved result files of a reduced protocol.
      </Alert>

      <Reveal as="section" className="section">
        <div className="grid g4">
          <div className="card glass"><Kpi value={dawn?.n_valid_samples ? fmtNum(dawn.n_valid_samples) : '—'} label="DAWN images scanned" /></div>
          <div className="card glass"><Kpi value={model.data?.detector ? `${(model.data.detector.params / 1e6).toFixed(1)} M` : model.data?.weights_exist ? 'on disk' : '—'} label="Detector parameters" /></div>
          <div className="card glass"><Kpi value="1,028" label="Gate parameters (Eq. 6)" title="1024 feature + 3 cue weights + bias" /></div>
          <div className="card glass"><Kpi value={trained.length || '0'} label="Trained front-end runs (excl. smoke)" /></div>
        </div>
      </Reveal>

      <Reveal as="section" className="section">
        <div className="grid g2">
          <div className="card hover-lift">
            <h3><ShieldAlert size={18} color="var(--warn)" /> Problem statement</h3>
            <p>{meta?.problem ?? '…'}</p>
            <div className="row">
              <span className="pill">ERUP-YOLO DAWN-sand: 27.82</span>
              <span className="pill">YOLOv3 + same aug: 31.18</span>
              <span className="pill bad">Δ −3.36 mAP@0.5 [paper, quoting ERUP-YOLO]</span>
            </div>
          </div>
          <div className="card hover-lift">
            <h3><Brain size={18} color="var(--cyan)" /> Abstract (summary)</h3>
            <p style={{ maxHeight: 220, overflow: 'auto' }}>{meta?.abstract ?? '…'}</p>
          </div>
        </div>
      </Reveal>

      <section className="section">
        <Reveal><h2>Proposed solution</h2><p>Three additions on top of ERUP-YOLO's BPW and KBL filters. With G ≡ 1 the pipeline is exactly ERUP-YOLO.</p></Reveal>
        <div className="grid g3">
          {[
            { icon: Layers, t: 'Low-resolution spatial gate', d: 'g = σ(W_F F + W_C C + b) on an 8×8 grid, bilinearly upsampled; blends the BPW output with the original image per region.' },
            { icon: Gauge, t: 'Saturation-aware penalty', d: 'One-sided L_sat penalises brightening of already-bright regions and pushing pixels past τ_hi — never brightening dark regions.' },
            { icon: Sparkles, t: 'Spatially heterogeneous augmentation', d: 'Mixes a brightening and a darkening curve under a smooth mask, so the gate has something spatial to learn.' },
          ].map(({ icon: Icon, t, d }, i) => (
            <Reveal key={t} delay={i * 90}><div className="card hover-lift" style={{ height: '100%' }}><h3><Icon size={18} color="var(--violet)" /> {t}</h3><p>{d}</p></div></Reveal>
          ))}
        </div>
      </section>

      <Reveal as="section" className="section">
        <h2>Model architecture</h2>
        <p className="small"><span className="pill" style={{ borderColor: 'rgba(59,130,246,.5)' }}>blue: ERUP-YOLO (prior work)</span>{' '}
          <span className="pill" style={{ borderColor: 'rgba(251,146,60,.55)' }}>orange: proposed in the paper</span></p>
        <div className="card">
          <div className="pipeline" role="list">
            {PIPE.map((n, i) => (
              <div key={n.t} style={{ display: 'contents' }}>
                <div className={`pipe-node ${n.k}`} role="listitem"><b>{n.t}</b><span>{n.s}</span></div>
                {i < PIPE.length - 1 && <div className="pipe-arrow" aria-hidden><div className="flow-dot" /></div>}
              </div>
            ))}
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Training: L = L_det + λ_s·L_sat + λ_tv·L_tv. The gate reads the encoder's last feature map F (1024×h×w) and
            three cue maps pooled to the same grid. Only the BPW stage is gated by default (gating KBL is ablation E2-g).
          </p>
        </div>
      </Reveal>

      <section className="section">
        <Reveal><h2>Research gaps examined</h2></Reveal>
        <div className="grid g3">
          {(meta?.gaps ?? []).map((g, i) => (
            <Reveal key={g.gap} delay={i * 60}>
              <div className="card hover-lift" style={{ height: '100%' }}>
                <h3 style={{ fontSize: '1rem' }}>{g.gap}</h3>
                <span className={`pill ${g.status.startsWith('Established') ? 'good' : g.status === 'Supported' ? 'info' : 'warn'}`}>{g.status}</span>
                <p className="small" style={{ marginTop: 10 }}>{g.closes}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <Reveal as="section" className="section">
        <div className="split">
          <div className="card">
            <h3><Database size={18} color="var(--cyan)" /> Dataset overview</h3>
            <div className="table-wrap">
              <table className="tbl">
                <thead><tr><th>Dataset</th><th>Role</th><th>Status</th><th className="num">Images</th></tr></thead>
                <tbody>
                  {(ds.data?.datasets ?? []).map((d) => {
                    const reps = Object.values(d.reports).filter(Boolean)
                    const n = reps.reduce((a, r) => a + (r?.n_valid_samples ?? 0), 0)
                    return (
                      <tr key={d.name}>
                        <td>{d.title}</td><td className="muted">{d.role}</td>
                        <td>{d.available ? <span className="pill good">on disk</span> : <span className="pill">not downloaded</span>}</td>
                        <td className="num">{d.available && reps.length ? fmtNum(n) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {ds.error && <Alert kind="error">{ds.error}</Alert>}
            <Link to="/datasets" className="small">Open Dataset Explorer →</Link>
          </div>
          <div className="card">
            <h3><Beaker size={18} color="var(--violet)" /> Project status (live)</h3>
            <div className="stack">
              {status.map((s) => (
                <div key={s.label} className="row" style={{ alignItems: 'flex-start', flexWrap: 'nowrap' }}>
                  {s.done === null ? <CircleDashed size={18} color="var(--text-3)" /> : s.done ? <CheckCircle2 size={18} color="var(--good)" /> : <CircleDashed size={18} color="var(--warn)" />}
                  <div><div style={{ color: 'var(--text)', fontSize: '.9rem' }}>{s.label}</div><div className="tiny muted">{s.note}</div></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Reveal>
    </>
  )
}
