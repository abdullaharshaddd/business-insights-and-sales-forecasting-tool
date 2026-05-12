import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts'
import api from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

const SEGMENT_COLOR = {
  'Champions':   { color: '#10B981', bg: 'rgba(16,185,129,0.12)', badge: 'badge-success' },
  'High-Value':  { color: '#6366F1', bg: 'rgba(99,102,241,0.12)', badge: 'badge-info' },
  'Mid-Value':   { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)', badge: 'badge-warning' },
  'Low-Value':   { color: '#EF4444', bg: 'rgba(239,68,68,0.12)',  badge: 'badge-danger' },
}

function getRiskLevel(rate) {
  if (rate < 0.45) return { label: 'Low Risk',    cls: 'badge-success' }
  if (rate < 0.65) return { label: 'Medium Risk',  cls: 'badge-warning' }
  if (rate < 0.80) return { label: 'High Risk',    cls: 'badge-danger' }
  return                   { label: 'Critical',    cls: 'badge-danger' }
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div style={{
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px', fontSize: 12,
      }}>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.fill, fontWeight: 600 }}>
            {p.name}: {(p.value * 100)?.toFixed ? `${(p.value * 100).toFixed(1)}%` : p.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

export default function Churn() {
  const [segments, setSegments]   = useState(null)
  const [models, setModels]       = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([api.get('/churn/segments'), api.get('/churn/models')])
      .then(([segRes, modRes]) => {
        // Backend returns { status, source, segments } or { status, source, models }
        setSegments(segRes.segments || [])
        setModels(modRes.models || [])
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const segBarData = (segments || []).map(s => ({
    segment: s.segment,
    churn_rate: s.churn_rate,
    auc_roc: s.auc_roc,
  }))

  const modelBarData = (models || []).map(m => ({
    model: m.model.replace('(Best)', '').trim(),
    'AUC-ROC': m.auc_roc,
    'F1':      m.f1,
    'Recall':  m.recall,
  }))

  return (
    <div className="fade-up">
      <PageHeader
        title="Churn Prediction"
        subtitle="DNN trained on RFM features · Online Retail dataset (2010-2011) · SMOTE oversampling"
      >
        <span className="badge badge-info">DNN + 4 Baselines</span>
      </PageHeader>

      {error && <div className="error-banner section-gap">⚠️ {error}</div>}

      {loading ? (
        <LoadingSpinner message="Loading churn evaluation data..." />
      ) : (
        <>
          {/* Segment Cards */}
          <div className="grid-auto section-gap">
            {(segments || []).map((seg, i) => {
              const cfg = SEGMENT_COLOR[seg.segment] || { color: 'var(--text-secondary)', bg: 'var(--bg-elevated)', badge: 'badge-neutral' }
              const risk = getRiskLevel(seg.churn_rate)
              const pct  = (seg.churn_rate * 100).toFixed(1)

              return (
                <div key={i} className="kpi-card" style={{ borderColor: `${cfg.color}30` }}>
                  <div className="kpi-card-header">
                    <span className="kpi-label">{seg.segment}</span>
                    <span className={`badge ${risk.cls}`}>{risk.label}</span>
                  </div>

                  <div className="kpi-value" style={{ color: cfg.color, fontSize: 32 }}>
                    {pct}%
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14 }}>
                    churn rate
                  </div>

                  {/* Risk bar */}
                  <div className="risk-bar-track">
                    <div
                      className="risk-bar-fill"
                      style={{ width: `${pct}%`, background: cfg.color }}
                    />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 14, fontSize: 12 }}>
                    <div>
                      <div style={{ color: 'var(--text-muted)' }}>AUC-ROC</div>
                      <div style={{ fontWeight: 700, color: cfg.color }}>{seg.auc_roc.toFixed(4)}</div>
                    </div>
                    {seg.f1 != null && (
                      <div>
                        <div style={{ color: 'var(--text-muted)' }}>F1 Score</div>
                        <div style={{ fontWeight: 700 }}>{seg.f1.toFixed(3)}</div>
                      </div>
                    )}
                    {seg.n_samples > 0 && (
                      <div>
                        <div style={{ color: 'var(--text-muted)' }}>Customers</div>
                        <div style={{ fontWeight: 700 }}>{seg.n_samples}</div>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Charts Row */}
          <div className="grid-2 section-gap">
            {/* Churn Rate by Segment */}
            <div className="card">
              <div className="card-header">
                <div>
                  <p className="card-title">Churn Rate by Segment</p>
                  <p className="card-subtitle">% customers with no purchase in last 30 days</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={segBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis dataKey="segment" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="churn_rate" name="Churn Rate" radius={[4,4,0,0]}>
                    {segBarData.map((entry, index) => {
                      const cfg = SEGMENT_COLOR[entry.segment] || {}
                      return <Cell key={index} fill={cfg.color || '#6366F1'} />
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Model Comparison */}
            <div className="card">
              <div className="card-header">
                <div>
                  <p className="card-title">Model Comparison</p>
                  <p className="card-subtitle">DNN vs Baselines · AUC-ROC / F1 / Recall</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={modelBarData} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)}
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" width={110}
                    tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                  <Tooltip formatter={v => v.toFixed(3)} contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-secondary)' }} />
                  <Bar dataKey="AUC-ROC" fill="#6366F1" radius={[0,3,3,0]} />
                  <Bar dataKey="F1"      fill="#06B6D4" radius={[0,3,3,0]} />
                  <Bar dataKey="Recall"  fill="#10B981" radius={[0,3,3,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Architecture Info */}
          <div className="card">
            <div className="card-header">
              <p className="card-title">DNN Architecture & Training Configuration</p>
              <span className="badge badge-success">✅ Trained</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              {[
                { title: 'Architecture', items: ['Input → Dense(128, ReLU)', 'BatchNorm → Dropout(0.3)', 'Dense(64, ReLU) → BN', 'Dense(32, ReLU) → BN', 'Dense(1, Sigmoid)'] },
                { title: 'Training Setup', items: ['Split: 70 / 15 / 15 stratified', 'SMOTE on training set only', 'HP Grid: 12 random samples', 'Best LR: 0.001 · Batch: 64', 'Epochs: 50 · Loss: BCE'] },
                { title: 'Features (11)', items: ['Recency, Frequency, Monetary', 'Avg Basket Size, Product Variety', 'Avg Unit Price, Country Enc', 'R/F/M Quintile Scores (1-5)', 'RFM Combined Score (3-15)'] },
              ].map((col, i) => (
                <div key={i} style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', padding: 16, border: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-accent)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 12 }}>
                    {col.title}
                  </p>
                  {col.items.map((item, j) => (
                    <div key={j} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '4px 0', borderBottom: '1px solid rgba(99,102,241,0.06)' }}>
                      {item}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
