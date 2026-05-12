import { useState, useEffect } from 'react'
import api from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

const TOPIC_OPTIONS = [
  { value: 'general',   label: 'General (Full Analysis)' },
  { value: 'revenue',   label: 'Revenue & Sales' },
  { value: 'delivery',  label: 'Delivery & Logistics' },
  { value: 'customer',  label: 'Customer & Retention' },
]

export default function Analytics() {
  const [tools, setTools]       = useState([])
  const [selected, setSelected] = useState('')
  const [topic, setTopic]       = useState('general')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [fetching, setFetching] = useState(true)
  const [error, setError]       = useState(null)
  const [history, setHistory]   = useState([])

  // Fetch tool list
  useEffect(() => {
    api.get('/analytics/tools')
      .then(data => {
        // FastAPI returns { status, tools, count } (interceptor returns response.data)
        setTools(data.tools || [])
        if (data.tools?.length) setSelected(data.tools[0].id)
      })
      .catch(err => setError(err.message))
      .finally(() => setFetching(false))
  }, [])

  const runTool = async () => {
    if (!selected || loading) return
    setLoading(true); setError(null); setResult(null)

    try {
      const res = await api.post('/analytics/run', {
        tool_id: selected,
        topic: selected === 'investigate_root_causes' ? topic : 'general',
      })
      setResult(res)
      setHistory(prev => [{ tool_id: selected, time: new Date().toLocaleTimeString(), result: res }, ...prev].slice(0, 5))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const selectedTool = tools.find(t => t.id === selected)

  const highlightText = (text) => {
    if (!text) return text
    // Highlight numbers and special markers
    return text
      .replace(/R\$[\d,]+\.?\d*/g, match => `\u001b[green]${match}`)
      .replace(/\d+\.\d+%/g, match => `\u001b[yellow]${match}`)
  }

  return (
    <div className="fade-up">
      <PageHeader
        title="Analytics Explorer"
        subtitle="Run any of the 11 deterministic analytical tools against the Olist marketplace database"
      >
        <span className="badge badge-info">11 Tools Available</span>
      </PageHeader>

      {fetching ? (
        <LoadingSpinner message="Loading analytical tools..." />
      ) : (
        <div className="grid-2" style={{ alignItems: 'start' }}>
          {/* Left: Tool selector */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Tool List */}
            <div className="card">
              <p className="card-title" style={{ marginBottom: 16 }}>Select Tool</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {tools.map(tool => (
                  <button
                    key={tool.id}
                    onClick={() => setSelected(tool.id)}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4,
                      padding: '12px 14px',
                      borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${selected === tool.id ? 'var(--border-bright)' : 'transparent'}`,
                      background: selected === tool.id ? 'rgba(99,102,241,0.12)' : 'transparent',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <span style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: selected === tool.id ? 'var(--indigo-light)' : 'var(--text-secondary)',
                      fontFamily: 'JetBrains Mono',
                    }}>
                      {tool.id}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {tool.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Topic selector (only for root causes) */}
            {selected === 'investigate_root_causes' && (
              <div className="card">
                <p className="card-title" style={{ marginBottom: 12 }}>Investigation Topic</p>
                <select value={topic} onChange={e => setTopic(e.target.value)}>
                  {TOPIC_OPTIONS.map(t => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Run button */}
            <button
              id="run-tool-btn"
              className="btn btn-primary"
              onClick={runTool}
              disabled={!selected || loading}
              style={{ width: '100%', justifyContent: 'center', padding: '14px' }}
            >
              {loading ? '⏳ Running Analysis...' : `▶ Run ${selectedTool?.id || 'Tool'}`}
            </button>

            {/* Recent History */}
            {history.length > 0 && (
              <div className="card">
                <p className="card-title" style={{ marginBottom: 12 }}>Recent Runs</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {history.map((h, i) => (
                    <button
                      key={i}
                      onClick={() => setResult(h.result)}
                      style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '8px 12px', background: 'var(--bg-elevated)',
                        border: '1px solid var(--border)', borderRadius: 8, cursor: 'pointer',
                        transition: 'border-color 0.15s',
                      }}
                    >
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}>
                        {h.tool_id}
                      </span>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{h.time}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Tool info card */}
            {selectedTool && (
              <div className="card" style={{ padding: '16px 20px' }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <div style={{
                    width: 40, height: 40, borderRadius: 8,
                    background: 'rgba(99,102,241,0.15)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 18, flexShrink: 0,
                  }}>🔬</div>
                  <div>
                    <p style={{ fontSize: 14, fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--indigo-light)' }}>
                      {selectedTool.id}
                    </p>
                    <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
                      {selectedTool.description}
                    </p>
                    {selectedTool.topics?.length > 0 && (
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                        {selectedTool.topics.map(t => (
                          <span key={t} className="badge badge-neutral">{t}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {error && <div className="error-banner">⚠️ {error}</div>}

            {/* Loading */}
            {loading && (
              <div className="card">
                <LoadingSpinner message="Querying Olist database & running analysis..." />
              </div>
            )}

            {/* Result */}
            {result && !loading && (
              <div className="card">
                <div className="card-header" style={{ marginBottom: 16 }}>
                  <div>
                    <p className="card-title">Analysis Result</p>
                    <p className="card-subtitle">{result.description}</p>
                  </div>
                  <button
                    className="btn btn-ghost"
                    style={{ fontSize: 12 }}
                    onClick={() => navigator.clipboard.writeText(result.result)}
                  >
                    📋 Copy
                  </button>
                </div>
                <div className="analytics-output">
                  {result.result}
                </div>
              </div>
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
              <div className="card">
                <div className="empty-state">
                  <div className="empty-icon">🔬</div>
                  <p className="empty-title">Select a tool and press Run</p>
                  <p className="empty-desc">Results will appear here in monospace format</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
