import { FlaskConical } from 'lucide-react'
import { useEffect, useState } from 'react'
import { postForm, type ProcessResponse, type Variant } from '../api'
import CompareSlider from '../components/CompareSlider'
import { Alert, Kpi, Loading, PageHead, Reveal } from '../components/ui'
import { useApi } from '../hooks'
import { Dropzone, useImageFile, VariantPicker } from './Studio'

const TILES: { key: string; title: string; eq: string; desc: string; heat?: boolean }[] = [
  { key: 'original', title: 'Input I', eq: 'letterboxed to 448', desc: 'Unprocessed image.' },
  { key: 'bpw', title: 'BPW output B(I)', eq: 'Eq. 1', desc: 'Global per-channel Bézier tone curve applied everywhere.' },
  { key: 'gate', title: 'Spatial gate G', eq: 'Eq. 6 + U(·)', desc: 'Where BPW is allowed (bright = 1 = use BPW, dark = 0 = keep original).', heat: true },
  { key: 'blended', title: 'Blended Ĩ', eq: 'Eq. 7', desc: 'G·B(I) + (1−G)·I — the image NSR/BRB are measured on.' },
  { key: 'final', title: 'Final Î = KBL(Ĩ)', eq: 'Eq. 8', desc: 'After the local 9×9 KBL filter; this is what the detector sees.' },
  { key: 'luminance', title: 'Local luminance μ', eq: 'Eq. 3', desc: 'Box-filtered Y = 0.27R + 0.67G + 0.06B.', heat: true },
  { key: 'dark_channel', title: 'Dark channel δ', eq: 'Eq. 4', desc: 'High in hazy or uniformly bright regions.', heat: true },
  { key: 'saturation', title: 'Near-saturated fraction ρ', eq: 'Eq. 5', desc: 'Local fraction of pixels with max channel > τ_hi.', heat: true },
  { key: 'annotated', title: 'Detections on Î', eq: 'YOLOv3u', desc: 'Real detector output at conf 0.25.' },
]

export default function Lab() {
  const variants = useApi<Variant[]>('/api/variants')
  const img = useImageFile()
  const [variant, setVariant] = useState('sg_full')
  const [checkpoint, setCheckpoint] = useState('')
  const [bpwSource, setBpwSource] = useState<'model' | 'manual'>('manual')
  const [strength, setStrength] = useState(0.6)
  const [gate, setGate] = useState('fixed')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<ProcessResponse | null>(null)

  useEffect(() => { if (!img.file) img.loadSample('Sand/sand_storm-151') }, []) // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    if (!img.file) return
    setBusy(true); setError(null)
    const fd = new FormData()
    fd.append('file', img.file); fd.append('variant', variant)
    if (checkpoint) fd.append('checkpoint', checkpoint)
    fd.append('bpw_source', bpwSource); fd.append('manual_strength', String(strength)); fd.append('gate_override', gate)
    try { setRes(await postForm<ProcessResponse>('/api/process', fd)) } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  const lowres = res?.gate_stats.lowres
  return (
    <>
      <PageHead eyebrow="SG-ERUP Processing Lab" title="See every stage of the front end">
        <p>Each tile is computed by an actual forward pass of the PyTorch implementation (Eqs. 1–8). The
          provenance banner says whether the BPW curve and gate come from a trained checkpoint, an untrained network,
          or a manual curve you set — manual curves are clearly illustrative, not model outputs.</p>
      </PageHead>

      <div className="split">
        <Reveal>
          <div className="card stack">
            <Dropzone onFile={img.accept} preview={img.preview} name={img.file?.name} />
            <div className="row small"><span className="muted">DAWN samples:</span>
              {[['Fog', 'Fog/foggy-001'], ['Rain', 'Rain/rain_storm-001'], ['Snow', 'Snow/snow_storm-001'], ['Sand', 'Sand/sand_storm-151']].map(([l, id]) =>
                <button key={id} className="btn sm" onClick={() => img.loadSample(id)}>{l}</button>)}
            </div>
            {img.fileError && <Alert kind="error">{img.fileError}</Alert>}
          </div>
        </Reveal>
        <Reveal delay={80}>
          <div className="card stack">
            {variants.data && <VariantPicker variants={variants.data} variant={variant} setVariant={setVariant} checkpoint={checkpoint} setCheckpoint={setCheckpoint} allowB0={false} />}
            <label className="field">BPW curve source
              <select value={bpwSource} onChange={(e) => setBpwSource(e.target.value as 'model' | 'manual')}>
                <option value="manual">Manual curve (illustrative)</option>
                <option value="model">Encoder prediction (checkpoint or untrained)</option>
              </select>
            </label>
            {bpwSource === 'manual' && (
              <label className="field">Manual curve strength: <b>{strength.toFixed(2)}</b> ({strength >= 0 ? 'brighten' : 'darken'})
                <input type="range" min={-1} max={1} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
              </label>
            )}
            <label className="field">Gate
              <select value={gate} onChange={(e) => setGate(e.target.value)}>
                <option value="none">Variant's own gate</option>
                <option value="fixed">B5 fixed gate σ(−(μ−τ_b)/s) — analytic, no training needed</option>
                <option value="one">G ≡ 1 (ERUP-YOLO)</option>
                <option value="zero">G ≡ 0 (KBL-only)</option>
                <option value="learned">Learned spatial gate</option>
              </select>
            </label>
            <button className="btn primary" onClick={run} disabled={!img.file || busy}><FlaskConical size={16} /> Process image</button>
            {busy && <div className="progress indeterminate"><div /></div>}
            {error && <Alert kind="error">{error}</Alert>}
          </div>
        </Reveal>
      </div>

      {busy && !res && <div className="section"><Loading label="Computing cue maps, gate and filters…" /></div>}
      {res && (
        <div className="section stack">
          <Alert kind={res.provenance.startsWith('Trained') ? 'info' : 'warn'}>
            <b>BPW parameters:</b> {res.provenance}. <b>Gate:</b> {res.gate_note}
          </Alert>
          <Reveal>
            <div className="grid g4">
              <div className="card"><Kpi value={`${(res.mechanism.nsr * 100).toFixed(3)}%`} label="NSR on Ĩ (Eq. 13)" /></div>
              <div className="card"><Kpi value={res.mechanism.brb.toFixed(4)} label="BRB on Ĩ (Eq. 14)" /></div>
              <div className="card"><Kpi value={`${(res.mechanism.nsr_final * 100).toFixed(3)}%`} label="NSR on Î (after KBL)" /></div>
              <div className="card"><Kpi value={res.gate_stats.mean.toFixed(3)} label={`Mean gate (min ${res.gate_stats.min.toFixed(2)}, max ${res.gate_stats.max.toFixed(2)})`} /></div>
            </div>
          </Reveal>
          <div className="split">
            <Reveal>
              <div className="card">
                <h3>Original vs. blended Ĩ</h3>
                <CompareSlider before={res.maps.original} after={res.maps.blended} beforeLabel="I" afterLabel="Ĩ = G·B(I) + (1−G)·I" />
              </div>
            </Reveal>
            <Reveal delay={60}>
              <div className="card">
                <h3>How the gate acts here</h3>
                <p className="small">The gate is computed on a low-resolution grid and bilinearly upsampled. Values near 1 let the
                  BPW curve through; values near 0 keep the original pixels. The saturation penalty (training only) pushes the gate
                  and curve to avoid brightening already-bright regions — measured here by NSR and BRB.</p>
                {lowres && (
                  <>
                    <div className="small muted" style={{ marginBottom: 6 }}>Low-resolution gate g ({lowres.length}×{lowres[0]?.length})</div>
                    <div className="gate-grid" style={{ gridTemplateColumns: `repeat(${lowres[0]?.length ?? 1}, 1fr)`, maxWidth: 260 }}>
                      {lowres.flat().map((v, i) => <div key={i} title={v.toFixed(3)} style={{ background: `rgba(34,211,238,${0.08 + 0.92 * v})` }} />)}
                    </div>
                  </>
                )}
                <div className="tiny muted" style={{ marginTop: 10 }}>τ_hi = {res.settings.tau_hi}, τ_b = {res.settings.tau_b}, s = {res.settings.s}, r = {res.settings.cue_radius} px</div>
              </div>
            </Reveal>
          </div>
          <div className="map-grid">
            {TILES.map((t, i) => (
              <Reveal key={t.key} delay={(i % 4) * 60}>
                <div className="map-tile hover-lift">
                  <img src={res.maps[t.key]} alt={t.title} />
                  <div className="cap"><b>{t.title} <span className="pill" style={{ marginLeft: 4 }}>{t.eq}</span></b>{t.desc}
                    {t.heat && <div className="colorbar" style={{ marginTop: 6 }} aria-label="colour scale 0 to 1" />}
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
          {res.theta_b && (
            <details className="card"><summary className="small">BPW parameters θ_B (12 values, [−1, 1])</summary>
              <pre className="mono small">{JSON.stringify(res.theta_b.map((v) => +v.toFixed(4)))}</pre></details>
          )}
        </div>
      )}
    </>
  )
}
