import { MoveHorizontal } from 'lucide-react'
import { useState, type ReactNode } from 'react'

/** Before/after slider. Keyboard accessible through a native range input. */
export default function CompareSlider({ before, after, beforeLabel = 'Original', afterLabel = 'Processed', overlay }: {
  before: string; after: string; beforeLabel?: string; afterLabel?: string; overlay?: ReactNode
}) {
  const [pos, setPos] = useState(50)
  return (
    <div className="compare">
      <img src={after} alt={afterLabel} />
      <div className="top" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={before} alt={beforeLabel} />
      </div>
      {overlay}
      <div className="handle" style={{ left: `${pos}%` }}>
        <div className="knob"><MoveHorizontal size={16} /></div>
      </div>
      <span className="tag" style={{ left: 10 }}>{beforeLabel}</span>
      <span className="tag" style={{ right: 10 }}>{afterLabel}</span>
      <input type="range" min={0} max={100} value={pos} aria-label="Comparison position"
        onChange={(e) => setPos(Number(e.target.value))} />
    </div>
  )
}
