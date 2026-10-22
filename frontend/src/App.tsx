import { AnimatePresence, motion, MotionConfig } from 'framer-motion'
import { CloudFog, CloudRain, CloudSnow, Menu, Sun, Wind } from 'lucide-react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import type { ModelStatus } from './api'
import AskAIWidget from './components/AskAI'
import { Loading } from './components/ui'
import WeatherBackground, { type Weather } from './components/WeatherBackground'
import { useApi, useReducedMotion } from './hooks'
import Home from './pages/Home'

const Studio = lazy(() => import('./pages/Studio'))
const Lab = lazy(() => import('./pages/Lab'))
const Analytics = lazy(() => import('./pages/Analytics'))
const Datasets = lazy(() => import('./pages/Datasets'))
const Experiments = lazy(() => import('./pages/Experiments'))
const Docs = lazy(() => import('./pages/Docs'))
const AskPage = lazy(() => import('./pages/AskPage'))

const LINKS = [
  ['/', 'Overview'], ['/studio', 'Detection Studio'], ['/lab', 'Processing Lab'], ['/analytics', 'Analytics'],
  ['/datasets', 'Datasets'], ['/experiments', 'Model & Experiments'], ['/docs', 'Docs'], ['/ask', 'Ask AI'],
] as const

const WEATHERS: { key: Weather; icon: typeof Sun; label: string }[] = [
  { key: 'fog', icon: CloudFog, label: 'Fog' }, { key: 'rain', icon: CloudRain, label: 'Rain' },
  { key: 'snow', icon: CloudSnow, label: 'Snow' }, { key: 'sand', icon: Wind, label: 'Sand' }, { key: 'clear', icon: Sun, label: 'Off' },
]

function Brand() {
  return (
    <NavLink to="/" className="brand" aria-label="SG-ERUP AI home">
      <img className="brand-mark" src="/favicon.svg" alt="" />
      <span>SG-ERUP AI <small>| Adverse Weather Intelligence</small></span>
    </NavLink>
  )
}

function ModelChip() {
  const { data, error } = useApi<ModelStatus>('/api/model/status', 15000)
  const cls = error ? 'bad' : data?.detector_loaded ? 'ok' : data?.detector_error ? 'bad' : 'warn'
  const text = error ? 'Backend offline' : data?.detector_loaded ? `Detector ready · ${data.detector?.device}` :
    data?.detector_error ? 'Detector error' : 'Loading detector…'
  return <span className="model-chip" title={data?.detector_error ?? error ?? data?.status_note ?? ''}><span className={`status-dot ${cls}`} />{text}</span>
}

export default function App() {
  const [booted, setBooted] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const [weather, setWeather] = useState<Weather>(() => {
    try { return (localStorage.getItem('sgerup-weather') as Weather) || 'fog' } catch { return 'fog' }
  })
  const loc = useLocation()
  const reduced = useReducedMotion()

  useEffect(() => {
    const t = setTimeout(() => setBooted(true), 650)
    return () => clearTimeout(t)
  }, [])
  useEffect(() => { setNavOpen(false); window.scrollTo({ top: 0 }) }, [loc.pathname])
  useEffect(() => { try { localStorage.setItem('sgerup-weather', weather) } catch { /* ignore */ } }, [weather])

  return (
    <MotionConfig reducedMotion="user">
      <div className={`loading-screen ${booted ? 'done' : ''}`} aria-hidden={booted}>
        <div className="loading-ring" />
        <div style={{ fontFamily: 'var(--display)', fontWeight: 700, letterSpacing: '.02em' }}>SG-ERUP AI</div>
        <div className="muted small">Initialising adverse-weather intelligence…</div>
      </div>
      <div className="bg-glow" aria-hidden />
      <div className="grid-overlay" aria-hidden />
      <WeatherBackground weather={weather} enabled={!reduced} />
      <div className="app">
        <a href="#main" className="sr-only">Skip to content</a>
        <header className="nav">
          <Brand />
          <button className="btn ghost sm nav-toggle" onClick={() => setNavOpen((o) => !o)} aria-label="Menu" aria-expanded={navOpen}><Menu size={18} /></button>
          <nav className={`nav-links ${navOpen ? 'open' : ''}`} aria-label="Main">
            {LINKS.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}
          </nav>
          <ModelChip />
        </header>
        <main id="main">
          <AnimatePresence mode="wait">
            <motion.div key={loc.pathname} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.28, ease: 'easeOut' }}>
              <Suspense fallback={<div style={{ padding: 40 }}><Loading label="Loading page…" /></div>}>
                <Routes location={loc}>
                  <Route path="/" element={<Home />} />
                  <Route path="/studio" element={<Studio />} />
                  <Route path="/lab" element={<Lab />} />
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/datasets" element={<Datasets />} />
                  <Route path="/experiments" element={<Experiments />} />
                  <Route path="/docs" element={<Docs />} />
                  <Route path="/ask" element={<AskPage />} />
                  <Route path="*" element={<div className="card"><h2>Page not found</h2><NavLink to="/">Back to overview</NavLink></div>} />
                </Routes>
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
        <footer>
          <div className="row" style={{ justifyContent: 'center', marginBottom: 8 }} role="group" aria-label="Background weather effect">
            <span className="tiny">Background:</span>
            {WEATHERS.map(({ key, icon: Icon, label }) => (
              <button key={key} className={`btn sm ${weather === key ? '' : 'ghost'}`} onClick={() => setWeather(key)} aria-pressed={weather === key}>
                <Icon size={14} /> {label}
              </button>
            ))}
          </div>
          SG-ERUP is a research hypothesis. Numbers shown on this site come only from saved experiment files. ·
          Research prototype built from the supplied preprint.
        </footer>
        {loc.pathname !== '/ask' && <AskAIWidget />}
      </div>
    </MotionConfig>
  )
}
