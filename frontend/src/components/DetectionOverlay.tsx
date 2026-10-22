import type { Detection } from '../api'

export const CLASS_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fbbf24', '#f87171', '#60a5fa', '#f472b6', '#facc15']
export const classColor = (id: number) => CLASS_COLORS[id % CLASS_COLORS.length]

/** SVG overlay of real detections in image pixel coordinates (scales with the image). */
export default function DetectionOverlay({ detections, width, height, hidden = false }: {
  detections: Detection[]; width: number; height: number; hidden?: boolean
}) {
  if (hidden) return null
  const fs = Math.max(11, Math.round(Math.max(width, height) / 60))
  const lw = Math.max(2, Math.round(Math.max(width, height) / 450))
  return (
    <svg className="overlay-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-hidden>
      {detections.map((d, i) => {
        const [x1, y1, x2, y2] = d.box
        const c = classColor(d.class_id)
        const label = `${d.class_name} ${(d.score * 100).toFixed(0)}%`
        return (
          <g key={i}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={c} strokeWidth={lw} rx={lw} />
            <rect x={x1} y={Math.max(0, y1 - fs - 6)} width={label.length * fs * 0.58 + 8} height={fs + 6} fill={c} rx={3} />
            <text x={x1 + 4} y={Math.max(fs, y1 - 5)} fontSize={fs} fill="#04101c" fontFamily="Inter, sans-serif" fontWeight={600}>{label}</text>
          </g>
        )
      })}
    </svg>
  )
}
