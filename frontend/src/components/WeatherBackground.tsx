import { useEffect, useRef } from 'react'
import { useReducedMotion } from '../hooks'

export type Weather = 'fog' | 'rain' | 'snow' | 'sand' | 'clear'

interface P { x: number; y: number; vx: number; vy: number; r: number; a: number }

const COUNTS: Record<Weather, number> = { rain: 140, snow: 110, sand: 160, fog: 14, clear: 0 }

/**
 * Lightweight canvas particles (decorative only). Pauses when the tab is hidden,
 * caps device pixel ratio, and renders nothing when the user prefers reduced motion.
 */
export default function WeatherBackground({ weather, enabled }: { weather: Weather; enabled: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || reduced || !enabled || weather === 'clear') {
      canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const ctx = canvas.getContext('2d')!
    let w = 0, h = 0, raf = 0
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
    const resize = () => {
      w = window.innerWidth; h = window.innerHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const n = Math.round(COUNTS[weather] * Math.min(1, (w * h) / (1440 * 900)))
    const spawn = (init: boolean): P => {
      const x = Math.random() * w, y = init ? Math.random() * h : -20
      switch (weather) {
        case 'rain': return { x, y, vx: -1.2, vy: 9 + Math.random() * 6, r: 10 + Math.random() * 12, a: 0.12 + Math.random() * 0.2 }
        case 'snow': return { x, y, vx: Math.random() - 0.5, vy: 0.4 + Math.random() * 1.1, r: 1 + Math.random() * 2.2, a: 0.25 + Math.random() * 0.45 }
        case 'sand': return { x: init ? x : -10, y: Math.random() * h, vx: 2.5 + Math.random() * 4, vy: (Math.random() - 0.5) * 0.8, r: 0.8 + Math.random() * 1.6, a: 0.15 + Math.random() * 0.35 }
        default: return { x, y: Math.random() * h, vx: 0.15 + Math.random() * 0.25, vy: 0, r: 160 + Math.random() * 220, a: 0.035 + Math.random() * 0.04 }
      }
    }
    const ps: P[] = Array.from({ length: n }, () => spawn(true))
    let t = 0
    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      t += 1
      for (const p of ps) {
        if (weather === 'snow') p.x += Math.sin((t + p.y) * 0.01) * 0.3
        p.x += p.vx; p.y += p.vy
        if (weather === 'rain') {
          ctx.strokeStyle = `rgba(147, 197, 253, ${p.a})`; ctx.lineWidth = 1
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + p.vx * 1.5, p.y - p.r); ctx.stroke()
        } else if (weather === 'fog') {
          const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r)
          g.addColorStop(0, `rgba(203, 213, 225, ${p.a})`); g.addColorStop(1, 'rgba(203,213,225,0)')
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill()
        } else {
          ctx.fillStyle = weather === 'sand' ? `rgba(251, 191, 120, ${p.a})` : `rgba(241, 245, 249, ${p.a})`
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill()
        }
        if (p.y > h + 30 || p.x > w + (weather === 'fog' ? 400 : 30) || p.x < -400) Object.assign(p, spawn(false), weather === 'fog' ? { x: -300 } : {})
      }
      raf = requestAnimationFrame(draw)
    }
    const onVis = () => { cancelAnimationFrame(raf); if (!document.hidden) raf = requestAnimationFrame(draw) }
    raf = requestAnimationFrame(draw)
    window.addEventListener('resize', resize)
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [weather, enabled, reduced])

  return <canvas ref={ref} className="bg-canvas" aria-hidden="true" />
}
