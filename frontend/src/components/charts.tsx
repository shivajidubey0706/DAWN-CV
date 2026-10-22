import {
  Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { ReactNode } from 'react'
import { useReducedMotion } from '../hooks'

// Fixed entity -> slot mapping (colour follows the variant, never its rank). Palette validated on the
// chart surface (#111a2e) with the dataviz validator: all checks pass in dark mode.
const SLOTS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
const VARIANT_SLOT: Record<string, number> = {
  b0_no_filter: 0, b2_erup: 1, b3_kbl_only: 2, b4_global_gate: 3, b5_fixed_gate: 4, sg_gate: 5, sg_gate_sha: 6, sg_full: 7,
}
export const OTHER_COLOR = '#8b93a7'
export const variantColor = (v: string) => (v in VARIANT_SLOT ? SLOTS[VARIANT_SLOT[v]] : OTHER_COLOR)
export const seriesColor = (i: number) => SLOTS[i % SLOTS.length]

const axis = { stroke: '#7d89a1', fontSize: 12, tickLine: false, axisLine: { stroke: 'rgba(148,163,184,.25)' } }
const grid = <CartesianGrid stroke="rgba(148,163,184,.10)" vertical={false} />

function TooltipBox({ active, payload, label, unit = '' }: {
  active?: boolean; payload?: { name: string; value: number; color: string; payload: Record<string, unknown> }[]; label?: string; unit?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'rgba(10,16,32,.96)', border: '1px solid rgba(148,163,184,.3)', borderRadius: 10, padding: '8px 11px', fontSize: 12.5 }}>
      <div style={{ color: '#e6ebf5', fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => {
        const sd = p.payload[`${p.name}__sd`] as number | undefined
        const n = p.payload[`${p.name}__n`] as number | undefined
        return (
          <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 7, color: '#a9b4c9' }}>
            <span style={{ width: 9, height: 9, borderRadius: 2, background: p.color }} />
            <span style={{ flex: 1 }}>{p.name}</span>
            <span style={{ color: '#e6ebf5', fontFamily: 'JetBrains Mono, monospace' }}>
              {p.value === null || p.value === undefined ? 'n/a' : `${Number(p.value).toFixed(2)}${unit}`}
              {sd ? ` ± ${sd.toFixed(2)}` : ''}{n ? ` (n=${n})` : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export interface GroupedDatum { group: string; [series: string]: number | string | null }

/** Grouped bars: x = group, one bar per series. Optional `${series}__sd` keys draw error bars. */
export function GroupedBars({ data, series, unit = '', height = 300, yLabel, colorOf, domain }: {
  data: GroupedDatum[]; series: string[]; unit?: string; height?: number; yLabel?: string
  colorOf?: (s: string, i: number) => string; domain?: [number, number | 'auto']
}) {
  const reduced = useReducedMotion()
  if (!data.length || !series.length) return <EmptyChart>No measured values for this selection.</EmptyChart>
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 4 }} barGap={2} barCategoryGap="22%">
        {grid}
        <XAxis dataKey="group" {...axis} />
        <YAxis {...axis} width={48} domain={domain ?? [0, 'auto']}
          label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#7d89a1', fontSize: 11, dx: 4, dy: 30 } : undefined} />
        <Tooltip cursor={{ fill: 'rgba(148,163,184,.07)' }} content={<TooltipBox unit={unit} />} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: '#a9b4c9' }} iconType="square" />}
        {series.map((s, i) => (
          <Bar key={s} dataKey={s} name={s} fill={colorOf ? colorOf(s, i) : seriesColor(i)} radius={[4, 4, 0, 0]}
            isAnimationActive={!reduced} animationDuration={900} animationBegin={i * 90} maxBarSize={44}>
            {data.some((d) => d[`${s}__sd`] != null) && <ErrorBar dataKey={`${s}__sd`} width={4} stroke="#cbd5e1" strokeWidth={1.2} />}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function HBars({ data, height, color = SLOTS[0], unit = '' }: {
  data: { name: string; value: number }[]; height?: number; color?: string; unit?: string
}) {
  const reduced = useReducedMotion()
  if (!data.length) return <EmptyChart>No data.</EmptyChart>
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(160, data.length * 34 + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="rgba(148,163,184,.10)" horizontal={false} />
        <XAxis type="number" {...axis} />
        <YAxis type="category" dataKey="name" {...axis} width={90} />
        <Tooltip cursor={{ fill: 'rgba(148,163,184,.07)' }} content={<TooltipBox unit={unit} />} />
        <Bar dataKey="value" name="count" fill={color} radius={[0, 4, 4, 0]} isAnimationActive={!reduced} animationDuration={900} maxBarSize={22} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function Lines({ data, x, series, height = 260, yLabel }: {
  data: Record<string, number | null>[]; x: string; series: { key: string; name: string; color: string; dashed?: boolean }[]
  height?: number; yLabel?: string
}) {
  const reduced = useReducedMotion()
  if (!data.length) return <EmptyChart>No history yet.</EmptyChart>
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 4 }}>
        {grid}
        <XAxis dataKey={x} {...axis} />
        <YAxis {...axis} width={52} domain={['auto', 'auto']}
          label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#7d89a1', fontSize: 11, dx: 4, dy: 30 } : undefined} />
        <Tooltip content={<TooltipBox />} cursor={{ stroke: 'rgba(148,163,184,.4)' }} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={2}
            strokeDasharray={s.dashed ? '5 4' : undefined} dot={data.length < 30 ? { r: 4 } : false}
            activeDot={{ r: 5 }} isAnimationActive={!reduced} animationDuration={1000} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function EmptyChart({ children }: { children: ReactNode }) {
  return (
    <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed rgba(148,163,184,.25)', borderRadius: 10 }}
      className="muted small">{children}</div>
  )
}
