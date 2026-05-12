import { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import api from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

const DAYS_OPTIONS = [7, 14, 30, 60, 90]

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    const yhat  = payload.find(p => p.dataKey === 'yhat')?.value
    const upper = payload.find(p => p.dataKey === 'yhat_upper')?.value
    const lower = payload.find(p => p.dataKey === 'yhat_lower')?.value
    return (
      <div style={{
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px', fontSize: 12,
      }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: 6 }}>📅 {label}</p>
        {yhat  != null && <p style={{ color: 'var(--indigo-light)',  fontWeight: 700 }}>Forecast: R${yhat.toLocaleString()}</p>}
        {upper != null && <p style={{ color: 'var(--text-muted)' }}>Upper CI: R${upper.toLocaleString()}</p>}
        {lower != null && <p style={{ color: 'var(--text-muted)' }}>Lower CI: R${lower.toLocaleString()}</p>}
      </div>
    )
  }
  return null
}

export default function Forecasting() {
  const [days, setDays]         = useState(30)
  const [data, setData]         = useState(null)
  const [metrics, setMetrics]   = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)

  const fetchAll = (d) => {
    setLoading(true); setError(null)
    Promise.all([
      api.get(`/forecast?days=${d}`),
      api.get('/forecast/metrics'),
    ])
      .then(([forecastRes, metricsRes]) => {
        setData(forecastRes)
        setMetrics(metricsRes.metrics)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchAll(days) }, [days])

  const chartData = (data?.points || []).map(p => ({
    date: p.date.slice(5),   // MM-DD
    yhat: Math.round(p.yhat),
    yhat_upper: Math.round(p.yhat_upper),
    yhat_lower: Math.max(0, Math.round(p.yhat_lower)),
    trend: Math.round(p.trend),
  }))

  const summary = data?.summary || {}

  return (
    <div className="fade-up">
      <PageHeader
        title="Sales Forecasting"
        subtitle="Prophet model trained on Online Retail daily revenue · UK Holidays · 95% Confidence Interval"
      >
        {/* Days selector */}
        <div className="tabs">
          {DAYS_OPTIONS.map(d => (
            <button
              key={d}
              className={`tab-btn ${days === d ? 'active' : ''}`}
              onClick={() => setDays(d)}
            >
              {d}d
            </button>
          ))}
        </div>
      </PageHeader>

      {error && (
        <div className="error-banner section-gap">
          ⚠️ {error}
        </div>
      )}

      {loading ? (
        <LoadingSpinner message="Loading Prophet forecast data..." />
      ) : (
        <>
          {/* Summary KPIs */}
          <div className="kpi-grid" style={{ marginBottom: 24 }}>
            {[
              { label: 'Projected Revenue', value: summary.total_revenue ? `R$${(summary.total_revenue/1000).toFixed(1)}k` : 'N/A', icon: '💰', color: 'var(--success)', bg: 'rgba(16,185,129,0.15)' },
              { label: 'Avg Daily Revenue', value: summary.avg_daily_revenue ? `R$${summary.avg_daily_revenue.toLocaleString()}` : 'N/A', icon: '📅', color: 'var(--indigo)', bg: 'rgba(99,102,241,0.15)' },
              { label: 'Trend Direction', value: summary.trend || 'N/A', icon: summary.trend === 'INCREASING' ? '📈' : '📉', color: summary.trend === 'INCREASING' ? 'var(--success)' : 'var(--danger)', bg: summary.trend === 'INCREASING' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)' },
              { label: 'Peak Revenue Day', value: summary.peak_day || 'N/A', icon: '🏆', color: 'var(--warning)', bg: 'rgba(245,158,11,0.15)' },
            ].map((k, i) => (
              <div key={i} className="kpi-card">
                <div className="kpi-card-header">
                  <span className="kpi-label">{k.label}</span>
                  <div className="kpi-icon" style={{ background: k.bg, color: k.color }}>{k.icon}</div>
                </div>
                <div className="kpi-value" style={{ fontSize: 20, color: k.color }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Main Chart */}
          <div className="card section-gap">
            <div className="card-header">
              <div>
                <p className="card-title">Revenue Forecast — Next {days} Days</p>
                <p className="card-subtitle">
                  Shaded area = 95% confidence interval · Blue line = point estimate · Source: {data?.source || 'prophet'}
                </p>
              </div>
              {data?.source === 'cached_csv' && (
                <span className="badge badge-warning">📁 Cached Data</span>
              )}
              {data?.source === 'live_prophet' && (
                <span className="badge badge-success">🔴 Live Model</span>
              )}
            </div>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={340}>
                <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#6366F1" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="#6366F1" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%"   stopColor="#6366F1" />
                      <stop offset="100%" stopColor="#06B6D4" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    interval={Math.floor(chartData.length / 7)}
                  />
                  <YAxis
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    tickFormatter={v => `R$${(v / 1000).toFixed(0)}k`}
                    width={70}
                    domain={['dataMin', 'dataMax']}
                    tickCount={8}
                    allowDecimals={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  {/* CI Band */}
                  <Area
                    type="monotone"
                    dataKey="yhat_upper"
                    stroke="none"
                    fill="url(#ciGradient)"
                    fillOpacity={1}
                    name="Upper CI"
                  />
                  <Area
                    type="monotone"
                    dataKey="yhat_lower"
                    stroke="none"
                    fill="var(--bg-base)"
                    fillOpacity={1}
                    name="Lower CI"
                  />
                  {/* Forecast Line */}
                  <Area
                    type="monotone"
                    dataKey="yhat"
                    stroke="url(#lineGradient)"
                    strokeWidth={2.5}
                    fill="none"
                    dot={false}
                    activeDot={{ r: 5, fill: 'var(--indigo)' }}
                    name="Forecast"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">📈</div>
                <p className="empty-title">No forecast data</p>
                <p className="empty-desc">Train Prophet: <code>python -m src.forecasting.prophet_model</code></p>
              </div>
            )}
          </div>

          {/* Model Metrics */}
          {metrics && (
            <div className="card">
              <div className="card-header">
                <div>
                  <p className="card-title">Model Performance — Cross-Validation Results</p>
                  <p className="card-subtitle">Initial: 180d · Period: 30d · Horizon: 30d</p>
                </div>
                <span className="badge badge-info">Prophet CV</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                {[
                  { label: 'RMSE',  value: metrics.rmse   ? `R$${Number(metrics.rmse).toLocaleString()}` : 'N/A', icon: '📐', color: 'var(--danger)' },
                  { label: 'MAE',   value: metrics.mae    ? `R$${Number(metrics.mae).toLocaleString()}`  : 'N/A', icon: '📏', color: 'var(--warning)' },
                  { label: 'MAPE',  value: metrics.mape   ? `${(Number(metrics.mape) * 100).toFixed(1)}%` : 'N/A', icon: '🎯', color: 'var(--indigo)' },
                  { label: 'CI Coverage', value: metrics.coverage ? `${(Number(metrics.coverage) * 100).toFixed(1)}%` : 'N/A', icon: '📊', color: 'var(--success)' },
                ].map((m, i) => (
                  <div key={i} style={{
                    background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)',
                    padding: '16px', border: '1px solid var(--border)', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 22, marginBottom: 8 }}>{m.icon}</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: m.color }}>{m.value}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{m.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
