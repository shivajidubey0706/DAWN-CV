import { Bot, Send, Sparkles, Trash2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { API_BASE } from '../api'
import { useApi } from '../hooks'
import { Markdown } from './Markdown'

interface Msg { role: 'user' | 'assistant'; content: string; mode?: string; sources?: { source: string; section: string; kind: string }[]; error?: string }
interface AskStatus { mode: 'llm' | 'fallback'; model: string | null; reason: string; n_chunks: number }

const SUGGESTIONS = [
  'How does the spatial gate decide where to apply the BPW curve?',
  'What do NSR and BRB measure?',
  'What did the short_real experiment measure on DAWN sand?',
  'Why is SHA needed to train the gate?',
  'What are the limitations of this implementation?',
  'How do I add the RTTS dataset?',
]

const STORAGE_KEY = 'sgerup-chat-v1'

function loadHistory(): Msg[] {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

/** Chat UI with SSE streaming from /api/ask. Works in LLM mode or local fallback mode. */
export function ChatBox({ compact = false }: { compact?: boolean }) {
  const status = useApi<AskStatus>('/api/ask/status')
  const [msgs, setMsgs] = useState<Msg[]>(loadHistory)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const bodyRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(msgs.slice(-40))) } catch { /* storage unavailable */ }
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs])

  useEffect(() => () => abortRef.current?.abort(), [])

  async function send(text: string) {
    const q = text.trim()
    if (!q || busy) return
    setInput('')
    const history: Msg[] = [...msgs, { role: 'user', content: q }]
    setMsgs([...history, { role: 'assistant', content: '' }])
    setBusy(true)
    const ctl = new AbortController()
    abortRef.current = ctl
    const update = (fn: (m: Msg) => Msg) => setMsgs((cur) => [...cur.slice(0, -1), fn(cur[cur.length - 1])])
    try {
      const res = await fetch(API_BASE + '/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctl.signal,
        body: JSON.stringify({ messages: history.filter((m) => m.content).map(({ role, content }) => ({ role, content })) }),
      })
      if (!res.ok || !res.body) {
        let detail = `${res.status}`
        try { detail = (await res.json()).detail ?? detail } catch { /* ignore */ }
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
      }
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const events = buf.split('\n\n')
        buf = events.pop() ?? ''
        for (const ev of events) {
          const line = ev.split('\n').find((l) => l.startsWith('data: '))
          if (!line) continue
          const e = JSON.parse(line.slice(6))
          if (e.type === 'meta') update((m) => ({ ...m, mode: e.mode, sources: e.sources }))
          else if (e.type === 'delta') update((m) => ({ ...m, content: m.content + e.text }))
          else if (e.type === 'error') update((m) => ({ ...m, error: e.message }))
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        const msg = (e as Error).message.includes('Failed to fetch') ? 'Cannot reach the backend (is it running on port 8000?)' : (e as Error).message
        update((m) => ({ ...m, error: msg }))
      }
    } finally {
      setBusy(false)
    }
  }

  const mode = status.data?.mode
  return (
    <>
      <div className="chat-body" ref={bodyRef} aria-live="polite">
        {status.error && <div className="alert error small">Ask AI backend unavailable: {status.error}</div>}
        {!msgs.length && (
          <div className="stack">
            <p className="small" style={{ margin: 0 }}>
              Ask about the SG-ERUP method, datasets, metrics, measured results or setup.{' '}
              {mode === 'fallback' && <>Running in <b>local fallback mode</b> (no LLM key on the server): answers are ranked excerpts from the paper, docs and result files.</>}
              {mode === 'llm' && <>Connected to <b>{status.data?.model}</b> with retrieval over the paper, docs and measured results.</>}
            </p>
            <div className="suggest">
              {SUGGESTIONS.slice(0, compact ? 4 : 6).map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
            </div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.role === 'assistant' && m.mode && (
              <div className="row" style={{ marginBottom: 6 }}>
                <span className={`pill ${m.mode === 'llm' ? 'info' : 'warn'}`}>{m.mode === 'llm' ? 'LLM answer' : 'Local fallback mode'}</span>
              </div>
            )}
            {m.content ? <Markdown text={m.content} /> : !m.error && <div className="typing" aria-label="Thinking"><span /><span /><span /></div>}
            {m.error && <div className="alert error small" style={{ marginTop: 6 }}>{m.error}</div>}
            {m.role === 'assistant' && !!m.sources?.length && m.content && (
              <div className="tiny muted" style={{ marginTop: 8 }}>
                Sources: {m.sources.slice(0, 5).map((s) => `${s.kind === 'measured' ? '📊' : '📄'} ${s.source} › ${s.section}`).join(' · ')}
              </div>
            )}
          </div>
        ))}
      </div>
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(input) }}>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask SG-ERUP AI…" rows={1} aria-label="Message"
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }} />
        <button className="btn primary" type="submit" disabled={busy || !input.trim()} aria-label="Send">
          {busy ? <div className="spinner" /> : <Send size={16} />}
        </button>
        {!!msgs.length && (
          <button type="button" className="btn ghost" onClick={() => { abortRef.current?.abort(); setMsgs([]) }} aria-label="Clear conversation" title="Clear conversation">
            <Trash2 size={16} />
          </button>
        )}
      </form>
    </>
  )
}

export default function AskAIWidget() {
  const [open, setOpen] = useState(false)
  if (!open)
    return (
      <button className="btn primary chat-fab" onClick={() => setOpen(true)} aria-label="Open Ask SG-ERUP AI">
        <Sparkles size={17} /> Ask SG-ERUP AI
      </button>
    )
  return (
    <div className="chat-panel" role="dialog" aria-label="Ask SG-ERUP AI">
      <div className="chat-head">
        <Bot size={18} color="var(--cyan)" />
        <b style={{ fontFamily: 'var(--display)' }}>Ask SG-ERUP AI</b>
        <button className="btn ghost sm" style={{ marginLeft: 'auto' }} onClick={() => setOpen(false)} aria-label="Close"><X size={16} /></button>
      </div>
      <ChatBox compact />
    </div>
  )
}
