import { useState } from 'react'
import { Equation, Markdown } from '../components/Markdown'
import { Alert, Loading, PageHead, Reveal } from '../components/ui'
import { useApi } from '../hooks'

interface PaperResp {
  meta: { title: string; status: string; research_questions: string[]; hypotheses: string[]; gaps: { gap: string; status: string; closes: string }[];
    equations: { id: string; name: string; tex: string }[]; protocol: string[]; limitations: string[] } | null
  references: { id: number; text: string }[]
}

const TABS = [
  ['objectives', 'Objectives'], ['method', 'Methodology & equations'], ['gaps', 'Research gaps'], ['protocol', 'Experimental protocol'],
  ['limitations', 'Limitations'], ['impl', 'Implementation notes'], ['repro', 'Reproducibility'], ['testing', 'Testing report'], ['refs', 'References'],
] as const

function DocMd({ name }: { name: string }) {
  const d = useApi<{ markdown: string }>(`/api/docs/${name}`)
  if (d.error) return <Alert kind="error">{d.error}</Alert>
  if (!d.data) return <Loading />
  return <div className="prose"><Markdown text={d.data.markdown} /></div>
}

export default function Docs() {
  const p = useApi<PaperResp>('/api/paper')
  const [tab, setTab] = useState<string>('objectives')
  const m = p.data?.meta
  return (
    <>
      <PageHead eyebrow="Documentation" title="Method, protocol and reproducibility">
        <p>{m?.title}</p>
      </PageHead>
      <Reveal><div className="tabs" role="tablist">{TABS.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{l}</button>)}</div></Reveal>
      {p.error && <Alert kind="error">{p.error}</Alert>}
      <div className="card section" style={{ marginTop: 18 }}>
        {tab === 'objectives' && m && (
          <div className="prose">
            <Alert kind="warn">{m.status}</Alert>
            <h3>Research questions</h3><ul>{m.research_questions.map((q) => <li key={q}>{q}</li>)}</ul>
            <h3>Falsifiable hypotheses</h3><ul>{m.hypotheses.map((q) => <li key={q}>{q}</li>)}</ul>
            <p className="small">Each hypothesis is rejected if its inequality fails after Holm correction across the confirmatory family (paper Sec. V-E).</p>
          </div>
        )}
        {tab === 'method' && m && (
          <div className="stack">
            <p className="small">Tensor convention in the implementation: float32, N×C×H×W, RGB, values in [0, 1]. Code locations are listed in “Implementation notes”.</p>
            {m.equations.map((e) => (
              <div key={e.id} className="row" style={{ alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: 6, flexWrap: 'nowrap' }}>
                <span className="pill" style={{ minWidth: 64, justifyContent: 'center' }}>Eq. {e.id}</span>
                <div style={{ minWidth: 170 }} className="small">{e.name}</div>
                <div style={{ flex: 1, minWidth: 0 }}><Equation tex={e.tex} /></div>
              </div>
            ))}
          </div>
        )}
        {tab === 'gaps' && m && (
          <div className="table-wrap"><table className="tbl"><thead><tr><th>Gap</th><th>Status</th><th>What would close it</th></tr></thead>
            <tbody>{m.gaps.map((g) => <tr key={g.gap}><td>{g.gap}</td><td>{g.status}</td><td>{g.closes}</td></tr>)}</tbody></table></div>
        )}
        {tab === 'protocol' && m && <div className="prose"><ul>{m.protocol.map((x) => <li key={x}>{x}</li>)}</ul>
          <p>What was actually run in this repository differs (frozen detector, reduced budget, 2 seeds) — see Implementation notes §3.</p></div>}
        {tab === 'limitations' && m && <div className="prose"><h3>From the paper</h3><ul>{m.limitations.map((x) => <li key={x}>{x}</li>)}</ul>
          <h3>Of this implementation</h3><p>See “Implementation notes” → Assumptions and Deviations, and the Testing report.</p></div>}
        {tab === 'impl' && <DocMd name="method" />}
        {tab === 'repro' && <DocMd name="readme" />}
        {tab === 'testing' && <DocMd name="testing" />}
        {tab === 'refs' && (
          <ol className="small" style={{ paddingLeft: 26 }}>
            {(p.data?.references ?? []).map((r) => <li key={r.id} value={r.id} style={{ marginBottom: 6, color: 'var(--text-2)' }}>{r.text}</li>)}
          </ol>
        )}
        {!m && !p.error && <Loading />}
      </div>
    </>
  )
}
