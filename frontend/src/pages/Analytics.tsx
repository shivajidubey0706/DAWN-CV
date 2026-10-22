import { Download, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { url, type ExperimentFile, type ExperimentSummary, type Stat, type SummaryRow, type Variant } from '../api'
import { GroupedBars, variantColor, type GroupedDatum } from '../components/charts'
import { Alert, Loading, NotEvaluated, PageHead, Reveal, fmtPct } from '../components/ui'
import { useApi } from '../hooks'

type ExpListItem = { name: string; n_result_files: number; has_summary: boolean; generated_at: string | null }
interface ExpDetail { summary: ExperimentSummary; files: ExperimentFile[]; plots: string[] }
interface CostRow { variant: string; params_frontend: number; params_total: number; gflops_frontend: number | null; gflops_total: number | null;
  latency_fp32: { median_ms: number; p90_ms: number; fps: number; peak_mem_mb?: number }; latency_fp16: { median_ms: number; fps: number } | null }
interface Cost { status?: string; message?: string; rows?: CostRow[]; device?: string; imgsz?: number; flops_convention?: string; latency_protocol?: string; environment?: { gpu?: string } }

const SUBSET_ORDER = ['all', 'fog', 'rain', 'snow', 'sand']
const stat = (r: SummaryRow | undefined, k: string) => (r?.[k] as Stat | undefined)

function Cell({ s, digits = 2, scale = 100 }: { s?: Stat; digits?: number; scale?: number }) {
  if (!s || s.mean === null || s.mean === undefined) return <NotEvaluated />
  return <>{(s.mean * scale).toFixed(digits)}{s.std != null && <span className="muted"> ± {(s.std * scale).toFixed(digits)}</span>}</>
}

export default function Analytics() {
  const list = useApi<ExpListItem[]>('/api/experiments')
  const variants = useApi<Variant[]>('/api/variants')
  const cost = useApi<Cost>('/api/cost')
  const [exp, setExp] = useState<string>('')
  const [refresh, setRefresh] = useState(0)
  const detail = useApi<ExpDetail>(exp ? `/api/experiments/${exp}${refresh ? `?refresh=true&_=${refresh}` : ''}` : null)
  const [dataset, setDataset] = useState('dawn')
  const [selected, setSelected] = useState<string[]>([])

  useEffect(() => {
    if (!exp && list.data?.length) setExp(list.data.find((e) => e.name === 'short_real')?.name ?? list.data[0].name)
  }, [list.data, exp])

  const summ = detail.data?.summary
  const rows = useMemo(() => (summ?.rows ?? []).filter((r) => r.dataset === dataset), [summ, dataset])
  const evaluated = useMemo(() => [...new Set(rows.map((r) => r.variant))], [rows])
  useEffect(() => {
    const pref = ['b0_no_filter', 'b2_erup', 'b4_global_gate', 'sg_full'].filter((v) => evaluated.includes(v))
    setSelected(pref.length ? pref : evaluated.slice(0, 4))
  }, [evaluated.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps

  const label = (v: string) => variants.data?.find((x) => x.key === v)?.label ?? v
  const row = (v: string, sub: string) => rows.find((r) => r.variant === v && r.subset === sub)
  const subsets = SUBSET_ORDER.filter((s) => rows.some((r) => r.subset === s)).concat([...new Set(rows.map((r) => r.subset))].filter((s) => !SUBSET_ORDER.includes(s)))

  const mk = (groups: string[], getter: (v: string, g: string) => Stat | undefined, scale = 100): GroupedDatum[] =>
    groups.map((g) => {
      const d: GroupedDatum = { group: g }
      selected.forEach((v) => {
        const s = getter(v, g)
        d[label(v)] = s?.mean != null ? +(s.mean * scale).toFixed(3) : null
        d[`${label(v)}__sd`] = s?.std != null ? +(s.std * scale).toFixed(3) : null
        d[`${label(v)}__n`] = s?.n ?? null
      })
      return d
    })

  const series = selected.map(label)
  const colorOf = (s: string) => variantColor(selected[series.indexOf(s)] ?? '')
  const toggle = (v: string) => setSelected((cur) => cur.includes(v) ? cur.filter((x) => x !== v) : cur.length >= 4 ? cur : [...cur, v])
  const anyRow = rows.find((r) => r.subset === 'all')
  const files = (detail.data?.files ?? []).filter((f) => f.dataset === dataset)
  const settings = files[0]?.settings as Record<string, unknown> | undefined

  return (
    <>
      <PageHead eyebrow="Research Analytics" title="Measured results only">
        <p>Every number on this page is read from <code>results/experiments/&lt;experiment&gt;/*.json</code>, written by
          <code> scripts/run_ablation.py</code>. Variants without a result file show “Not evaluated”. Values are % (mean ± s.d. over seeds).</p>
      </PageHead>

      <Reveal>
        <div className="card row between">
          <div className="row">
            <label className="field" style={{ minWidth: 200 }}>Experiment
              <select value={exp} onChange={(e) => setExp(e.target.value)}>
                {(list.data ?? []).map((e) => <option key={e.name} value={e.name}>{e.name} ({e.n_result_files} files)</option>)}
              </select>
            </label>
            <label className="field" style={{ minWidth: 140 }}>Dataset
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {(summ?.datasets ?? ['dawn']).map((d) => <option key={d}>{d}</option>)}
              </select>
            </label>
          </div>
          <div className="row">
            <button className="btn sm ghost" onClick={() => setRefresh(Date.now())} title="Re-aggregate from result files"><RefreshCw size={14} /> Re-aggregate</button>
            {exp && ['csv', 'json', 'md'].map((f) => <a key={f} className="btn sm" href={url(`/api/experiments/${exp}/export?format=${f}`)}><Download size={14} /> {f.toUpperCase()}</a>)}
          </div>
        </div>
      </Reveal>

      {list.error && <Alert kind="error">{list.error}</Alert>}
      {list.data && !list.data.length && <Alert kind="info">No experiments yet. Run <code>python scripts/run_ablation.py --config configs/smoke.yaml</code>.</Alert>}
      {detail.loading && !summ && <div className="section"><Loading /></div>}
      {detail.error && <Alert kind="error">{detail.error}</Alert>}

      {summ && (
        <>
          <Reveal className="section">
            <div className="card">
              <h3>Evaluation settings</h3>
              <div className="row small">
                <span className="pill info">experiment: {summ.experiment}</span>
                <span className="pill">dataset: {dataset}</span>
                <span className="pill">images: {anyRow?.n_images ?? '—'}</span>
                <span className="pill">GT boxes: {anyRow?.n_gt ?? '—'}</span>
                <span className="pill">seeds: {summ.seeds.join(', ')}</span>
                <span className="pill">input: {String(settings?.imgsz ?? '—')}</span>
                <span className="pill">score thr (P/R): {String(settings?.conf_pr ?? '—')}</span>
                <span className="pill">classes: {(settings?.classes as string[] | undefined)?.join(', ') ?? '—'}</span>
                <span className="pill">generated {summ.generated_at}</span>
              </div>
              {summ.experiment === 'smoke' && <Alert kind="warn">The smoke experiment uses 40 images and 64 training images — a pipeline check, not research evidence.</Alert>}
              {summ.seeds.length < 8 && <p className="tiny muted" style={{ marginTop: 10 }}>{summ.seeds.length} seed(s) completed; the paper's protocol requires 8 paired seeds. With fewer than 6 pairs a two-sided Wilcoxon test cannot reach p &lt; 0.05, so all comparisons are inconclusive.</p>}
            </div>
          </Reveal>

          <Reveal>
            <div className="card">
              <h3>Variants in charts (max 4, colour fixed per variant)</h3>
              <div className="row">
                {evaluated.map((v) => (
                  <button key={v} className={`btn sm ${selected.includes(v) ? '' : 'ghost'}`} onClick={() => toggle(v)} aria-pressed={selected.includes(v)}>
                    <span className="swatch" style={{ background: variantColor(v) }} /> {label(v)}
                  </button>
                ))}
              </div>
            </div>
          </Reveal>

          <div className="grid g2 section">
            <Reveal><div className="card"><h3>mAP@0.5 by weather subset</h3><div className="card-sub">VOC all-point AP, IoU 0.5; error bars = s.d. over seeds</div>
              <GroupedBars data={mk(subsets, (v, g) => stat(row(v, g), 'map50'))} series={series} colorOf={colorOf} unit="%" yLabel="mAP@0.5 (%)" /></div></Reveal>
            <Reveal delay={60}><div className="card"><h3>Precision / recall / mAP@[.5:.95]</h3><div className="card-sub">Overall; P/R at score ≥ 0.25, IoU 0.5</div>
              <GroupedBars data={mk(['precision', 'recall', 'map50_95'], (v, g) => stat(row(v, 'all'), g))} series={series} colorOf={colorOf} unit="%" yLabel="%" /></div></Reveal>
            <Reveal><div className="card"><h3>Recall by object brightness tertile</h3><div className="card-sub">GT boxes split by mean input luminance (H3 looks at the bright tertile)</div>
              <GroupedBars data={mk(['dark', 'medium', 'bright'], (v, g) => stat(row(v, 'all'), `recall_${g}`))} series={series} colorOf={colorOf} unit="%" yLabel="recall (%)" /></div></Reveal>
            <Reveal delay={60}><div className="card"><h3>AP by object size (COCO areas)</h3><div className="card-sub">mAP@[.5:.95] for small / medium / large boxes</div>
              <GroupedBars data={mk(['small', 'medium', 'large'], (v, g) => stat(row(v, 'all'), `ap_${g}`))} series={series} colorOf={colorOf} unit="%" yLabel="AP (%)" /></div></Reveal>
            <Reveal><div className="card"><h3>Newly-saturated ratio (NSR, Eq. 13)</h3><div className="card-sub">% of pixels newly pushed above τ_hi in Ĩ; lower = less over-brightening</div>
              <GroupedBars data={mk(subsets, (v, g) => stat(row(v, g), 'nsr'))} series={series} colorOf={colorOf} unit="%" yLabel="NSR (%)" /></div></Reveal>
            <Reveal delay={60}><div className="card"><h3>Bright-region brightening (BRB, Eq. 14)</h3><div className="card-sub">β-weighted mean luminance increase in Ĩ (×100)</div>
              <GroupedBars data={mk(subsets, (v, g) => stat(row(v, g), 'brb'))} series={series} colorOf={colorOf} yLabel="BRB ×100" /></div></Reveal>
          </div>

          <Reveal className="section">
            <div className="card">
              <h3>All variants · {dataset}</h3>
              <div className="table-wrap" style={{ maxHeight: 560 }}>
                <table className="tbl">
                  <thead><tr><th>Variant</th><th>Subset</th><th className="num">Seeds</th><th className="num">Images</th><th className="num">mAP@0.5</th><th className="num">mAP@[.5:.95]</th><th className="num">P</th><th className="num">R</th><th className="num">R bright</th><th className="num">R dark</th><th className="num">NSR</th><th className="num">BRB×100</th></tr></thead>
                  <tbody>
                    {(variants.data ?? []).flatMap((v) => {
                      const rs = subsets.map((s) => row(v.key, s)).filter(Boolean) as SummaryRow[]
                      if (!rs.length) return [<tr key={v.key}><td><span className="swatch" style={{ background: variantColor(v.key), display: 'inline-block', marginRight: 6 }} />{v.label}</td><td colSpan={11}><NotEvaluated /></td></tr>]
                      return rs.map((r) => (
                        <tr key={v.key + r.subset}>
                          <td>{r.subset === 'all' ? <><span className="swatch" style={{ background: variantColor(v.key), display: 'inline-block', marginRight: 6 }} />{v.label}</> : ''}</td>
                          <td>{r.subset}</td><td className="num">{r.n_seeds}</td><td className="num">{r.n_images}</td>
                          <td className="num"><Cell s={stat(r, 'map50')} /></td><td className="num"><Cell s={stat(r, 'map50_95')} /></td>
                          <td className="num"><Cell s={stat(r, 'precision')} /></td><td className="num"><Cell s={stat(r, 'recall')} /></td>
                          <td className="num"><Cell s={stat(r, 'recall_bright')} /></td><td className="num"><Cell s={stat(r, 'recall_dark')} /></td>
                          <td className="num"><Cell s={stat(r, 'nsr')} digits={3} /></td><td className="num"><Cell s={stat(r, 'brb')} digits={3} /></td>
                        </tr>
                      ))
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </Reveal>

          <Reveal>
            <div className="card">
              <h3>Paired differences vs {summ.reference} (mAP@0.5 points)</h3>
              <div className="card-sub">Mean paired difference over seeds, 95% t-interval, two-sided Wilcoxon signed-rank. Exploratory unless marked confirmatory.</div>
              <div className="table-wrap" style={{ maxHeight: 420 }}>
                <table className="tbl">
                  <thead><tr><th>Variant</th><th>Dataset / subset</th><th className="num">Pairs</th><th className="num">Δ mean</th><th className="num">95% CI</th><th className="num">Wilcoxon p</th><th>Note</th></tr></thead>
                  <tbody>
                    {summ.comparisons.filter((c) => c.dataset === dataset).map((c) => (
                      <tr key={c.variant + c.subset}>
                        <td>{label(c.variant)}</td><td>{c.dataset} / {c.subset}</td><td className="num">{c.n_pairs}</td>
                        <td className="num" style={{ color: (c.diff.mean ?? 0) > 0 ? 'var(--good)' : (c.diff.mean ?? 0) < 0 ? 'var(--bad)' : undefined }}>
                          {c.diff.mean != null ? `${(c.diff.mean * 100) > 0 ? '+' : ''}${(c.diff.mean * 100).toFixed(2)}` : '—'}</td>
                        <td className="num">{c.diff.ci ? `[${(c.diff.ci[0] * 100).toFixed(2)}, ${(c.diff.ci[1] * 100).toFixed(2)}]` : 'n/a'}</td>
                        <td className="num">{c.wilcoxon_p != null ? c.wilcoxon_p.toFixed(3) : 'n/a'}</td>
                        <td className="tiny muted">{c.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Reveal>

          <Reveal className="section">
            <div className="card">
              <h3>Per-class AP@0.5 (seed-level files)</h3>
              <div className="table-wrap" style={{ maxHeight: 420 }}>
                <table className="tbl">
                  <thead><tr><th>Variant</th><th className="num">Seed</th>{Object.keys(files[0]?.per_class ?? {}).map((c) => <th key={c} className="num">{c}</th>)}<th className="num">Bootstrap 95% CI (mAP@0.5)</th><th>Weights</th></tr></thead>
                  <tbody>
                    {files.map((f) => (
                      <tr key={f.file}>
                        <td>{label(f.variant)}</td><td className="num">{f.seed}</td>
                        {Object.entries(f.per_class ?? {}).map(([c, v]) => <td key={c} className="num">{v.ap50 != null ? fmtPct(v.ap50) : <NotEvaluated text="no GT" />}<span className="muted tiny"> /{v.n_gt}</span></td>)}
                        <td className="num">{f.bootstrap ? `[${fmtPct(f.bootstrap.ci95[0])}, ${fmtPct(f.bootstrap.ci95[1])}]` : '—'}</td>
                        <td className="tiny">{f.variant === 'b0_no_filter' ? 'detector only' : f.trained ? 'trained' : 'untrained'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Reveal>
        </>
      )}

      <Reveal className="section">
        <div className="card">
          <h3>Computational cost (E7)</h3>
          {cost.data?.rows ? (
            <>
              <div className="card-sub">{cost.data.environment?.gpu ?? cost.data.device} · {cost.data.imgsz}×{cost.data.imgsz}, batch 1 · {cost.data.latency_protocol}</div>
              <div className="grid g2">
                <GroupedBars data={cost.data.rows.map((r) => ({ group: r.variant, 'median latency fp32 (ms)': +r.latency_fp32.median_ms.toFixed(2) }))}
                  series={['median latency fp32 (ms)']} unit=" ms" yLabel="ms" height={240} />
                <GroupedBars data={cost.data.rows.map((r) => ({ group: r.variant, 'front-end params (M)': +(r.params_frontend / 1e6).toFixed(3) }))}
                  series={['front-end params (M)']} colorOf={() => '#9085e9'} yLabel="M params" height={240} />
              </div>
              <div className="table-wrap" style={{ marginTop: 12 }}>
                <table className="tbl"><thead><tr><th>Variant</th><th className="num">Params (front end)</th><th className="num">Params (total)</th><th className="num">GFLOPs (front end)</th><th className="num">GFLOPs (total)</th><th className="num">Median ms fp32</th><th className="num">p90 ms</th><th className="num">FPS</th><th className="num">ms fp16</th><th className="num">Peak MB</th></tr></thead>
                  <tbody>{cost.data.rows.map((r) => (
                    <tr key={r.variant}><td>{label(r.variant)}</td><td className="num">{r.params_frontend.toLocaleString()}</td><td className="num">{r.params_total.toLocaleString()}</td>
                      <td className="num">{r.gflops_frontend?.toFixed(2) ?? '—'}</td><td className="num">{r.gflops_total?.toFixed(2) ?? '—'}</td>
                      <td className="num">{r.latency_fp32.median_ms.toFixed(2)}</td><td className="num">{r.latency_fp32.p90_ms.toFixed(2)}</td><td className="num">{r.latency_fp32.fps.toFixed(1)}</td>
                      <td className="num">{r.latency_fp16?.median_ms.toFixed(2) ?? '—'}</td><td className="num">{r.latency_fp32.peak_mem_mb ?? '—'}</td></tr>))}</tbody></table>
              </div>
              <p className="tiny muted" style={{ marginTop: 8 }}>FLOPs convention: {cost.data.flops_convention}</p>
            </>
          ) : <p className="small"><NotEvaluated text="Not measured yet." /> {cost.data?.message}</p>}
        </div>
      </Reveal>
    </>
  )
}
