import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
import api from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

const KPI_CONFIG = {
  revenue: {
    label: 'Total Revenue',
    icon: '💰',
    color: 'var(--success)',
    bg: 'rgba(16,185,129,0.15)',
    format: (v) => `R$${(v / 1_000_000).toFixed(2)}M`,
  },
  aov: {
    label: 'Avg Order Value',
    icon: '🛒',
    color: 'var(--indigo)',
    bg: 'rgba(99,102,241,0.15)',
    format: (v) => `R$${v.toFixed(2)}`,
  },
  on_time_delivery_rate: {
    label: 'On-Time Delivery',
    icon: '🚚',
    color: 'var(--cyan)',
    bg: 'rgba(6,182,212,0.15)',
    format: (v) => `${v.toFixed(1)}%`,
  },
  avg_review_score: {
    label: 'Avg Review Score',
    icon: '⭐',
    color: 'var(--warning)',
    bg: 'rgba(245,158,11,0.15)',
    format: (v) => `${v.toFixed(2)} / 5`,
  },
  cancellation_rate: {
    label: 'Cancellation Rate',
    icon: '❌',
    color: 'var(--danger)',
    bg: 'rgba(239,68,68,0.15)',
    format: (v) => `${v.toFixed(2)}%`,
  },
  freight_ratio: {
    label: 'Freight-to-Price',
    icon: '📦',
    color: 'var(--violet)',
    bg: 'rgba(139,92,246,0.15)',
    format: (v) => `${v.toFixed(1)}%`,
  },
  avg_payment_installments: {
    label: 'Avg Installments',
    icon: '💳',
    color: 'var(--cyan-dark)',
    bg: 'rgba(8,145,178,0.15)',
    format: (v) => `${v.toFixed(1)}x`,
  },
}

const GEO_COLORS = ['#6366F1', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#64748B']

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 14px',
        fontSize: 13,
      }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.fill || p.stroke || 'var(--text-primary)', fontWeight: 600 }}>
            {p.name}: {typeof p.value === 'number' ? `R$${p.value.toLocaleString()}` : p.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

function KPICard({ id, data }) {
  const cfg = KPI_CONFIG[id]
  if (!cfg || data == null) return null
  const val = data.value

  return (
    <div className="kpi-card fade-up">
      <div className="kpi-card-header">
        <span className="kpi-label">{cfg.label}</span>
        <div className="kpi-icon" style={{ background: cfg.bg, color: cfg.color }}>
          {cfg.icon}
        </div>
      </div>
      {val !== null && val !== undefined ? (
        <div className="kpi-value" style={{ color: cfg.color }}>
          {cfg.format(val)}
        </div>
      ) : (
        <div className="kpi-value" style={{ color: 'var(--text-muted)', fontSize: 16 }}>
          Unavailable
        </div>
      )}
      {data.unit && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
          {data.unit}
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [kpis, setKpis]           = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [lastUpdated, setLast]    = useState(null)

  const fetchKPIs = () => {
    setLoading(true)
    setError(null)
    api.get('/dashboard/kpis')
      .then(data => {
        setKpis(data.data)
        setLast(new Date().toLocaleTimeString())
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchKPIs() }, [])

  const topCategories   = kpis?.top_categories   || []
  const geoDistribution = kpis?.geo_distribution || []

  const catChartData = topCategories.map(c => ({
    name: (c.category || c.product_category_name_english || 'Unknown').replace(/_/g, ' ').slice(0, 20),
    revenue: Math.round(c.revenue || 0),
  }))

  const geoPieData = geoDistribution.slice(0, 8).map((g, i) => ({
    name: g.state || g.customer_state || '?',
    value: Math.round(g.revenue || 0),
    fill: GEO_COLORS[i % GEO_COLORS.length],
  }))

  return (
    <div className="fade-up">
      <PageHeader
        title="Business Intelligence Overview"
        subtitle="Live KPIs from Olist Marketplace · Powered by the deterministic KPI Engine"
      >
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {lastUpdated && (
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Updated {lastUpdated}
            </span>
          )}
          <button className="btn btn-ghost" onClick={fetchKPIs} disabled={loading}>
            🔄 Refresh
          </button>
        </div>
      </PageHeader>

      {/* Error */}
      {error && (
        <div className="error-banner section-gap">
          ⚠️ {error} — Make sure the FastAPI server is running on port 8000.
        </div>
      )}

      {/* KPI Grid */}
      {loading ? (
        <LoadingSpinner message="Fetching live KPIs from Olist database..." />
      ) : (
        <>
          <div className="kpi-grid">
            {Object.keys(KPI_CONFIG).map(id => (
              <KPICard key={id} id={id} data={kpis?.[id]} />
            ))}
          </div>

          {/* Charts Row */}
          <div className="grid-2 section-gap">
            {/* Top Categories */}
            <div className="card">
              <div className="card-header">
                <div>
                  <p className="card-title">Top Categories by Revenue</p>
                  <p className="card-subtitle">Delivered orders only · R$ = price + freight</p>
                </div>
              </div>
              {catChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={catChartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" horizontal={false} />
                    <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                      tickFormatter={v => `R$${(v / 1000).toFixed(0)}k`} />
                    <YAxis type="category" dataKey="name" width={120}
                      tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="revenue" name="Revenue" fill="url(#barGradient)" radius={[0, 4, 4, 0]} />
                    <defs>
                      <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#6366F1" />
                        <stop offset="100%" stopColor="#06B6D4" />
                      </linearGradient>
                    </defs>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">📊</div>
                  <p className="empty-title">No category data available</p>
                  <p className="empty-desc">Ensure the Olist database is populated</p>
                </div>
              )}
            </div>

            {/* Geographic Distribution */}
            <div className="card">
              <div className="card-header">
                <div>
                  <p className="card-title">Revenue by State</p>
                  <p className="card-subtitle">Top 8 Brazilian states by marketplace revenue</p>
                </div>
              </div>
              {geoPieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie
                      data={geoPieData}
                      cx="50%" cy="50%"
                      outerRadius={90}
                      innerRadius={48}
                      paddingAngle={3}
                      dataKey="value"
                      nameKey="name"
                    >
                      {geoPieData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} stroke="transparent" />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(v) => [`R$${v.toLocaleString()}`, 'Revenue']}
                      contentStyle={{
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border)',
                        borderRadius: 8,
                      }}
                    />
                    <Legend
                      iconType="circle"
                      iconSize={8}
                      wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">🗺️</div>
                  <p className="empty-title">No geographic data</p>
                  <p className="empty-desc">Ensure the Olist database is populated</p>
                </div>
              )}
            </div>
          </div>

          {/* System Info Strip */}
          <div className="card" style={{ padding: '14px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 14 }}>🗄️</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Olist SQLite</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 14 }}>🤖</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>LangGraph Multi-Agent</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 14 }}>📡</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Groq API · Llama 3.3 70B</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 14 }}>🔮</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Prophet Forecasting · DNN Churn</span>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <span className="badge badge-info">v2.0</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
