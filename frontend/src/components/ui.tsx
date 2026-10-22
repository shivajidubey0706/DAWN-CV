import { AlertTriangle, Info, XCircle } from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'
import { useInView } from '../hooks'

export function Reveal({ children, delay = 0, className = '', as = 'div' }: {
  children: ReactNode; delay?: number; className?: string; as?: 'div' | 'section'
}) {
  const { ref, inView } = useInView<HTMLDivElement>()
  const Tag = as
  return (
    <Tag ref={ref} className={`reveal ${inView ? 'in' : ''} ${className}`} style={{ '--d': `${delay}ms` } as CSSProperties}>
      {children}
    </Tag>
  )
}

export function Alert({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'error'; children: ReactNode }) {
  const Icon = kind === 'error' ? XCircle : kind === 'warn' ? AlertTriangle : Info
  return (
    <div className={`alert ${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
      <Icon size={17} aria-hidden />
      <div>{children}</div>
    </div>
  )
}

export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <Reveal className="page-head">
      <div className="eyebrow">{eyebrow}</div>
      <h1 style={{ fontSize: 'clamp(1.7rem, 3.4vw, 2.5rem)' }}>{title}</h1>
      {children}
    </Reveal>
  )
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="row muted small" role="status"><div className="spinner" aria-hidden /> {label}</div>
  )
}

export function NotEvaluated({ text = 'Not evaluated' }: { text?: string }) {
  return <span className="not-eval">{text}</span>
}

export function Kpi({ value, label, title }: { value: ReactNode; label: string; title?: string }) {
  return (
    <div className="kpi" title={title}>
      <span className="v">{value}</span>
      <span className="l">{label}</span>
    </div>
  )
}

export const fmtPct = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? null : (v * 100).toFixed(digits)

export const fmtNum = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })

export function downloadText(name: string, text: string, mime = 'application/json') {
  const blob = new Blob([text], { type: mime })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
