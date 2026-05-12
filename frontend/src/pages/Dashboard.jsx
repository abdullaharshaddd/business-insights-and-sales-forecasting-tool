import { useState, useEffect } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend, Area, AreaChart
} from 'recharts'
import api from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

// ─── KPI Config — all from database ───────────────────────────────────────
const KPI_CONFIG = {
  // Revenue
  revenue:             { label: 'Total GMV',            icon: '💰', category: 'revenue',   format: v => `R$${(v / 1_000_000).toFixed(2)}M` },
  revenue_delivered:    { label: 'Delivered Revenue',   icon: '✅', category: 'revenue',   format: v => `R$${(v / 1_000_000).toFixed(2)}M` },
  total_revenue_retail: { label: 'Retail Revenue (UK)', icon: '🇬🇧', category: 'revenue',   format: v => `£${(v / 1_000_000).toFixed(2)}M` },
  aov:                  { label: 'Avg Order Value',    icon: '🛒', category: 'revenue',   format: v => `R$${v.toFixed(2)}` },
  aov_delivered:        { label: 'AOV (Delivered)',     icon: '📦', category: 'revenue',   format: v => `R$${v.toFixed(2)}` },

  // Orders
  total_orders:         { label: 'Total Orders',        icon: '📋', category: 'orders',    format: v => v?.toLocaleString() },
  delivered_orders:     { label: 'Delivered',          icon: '🚚', category: 'orders',    format: v => v?.toLocaleString() },
  pending_orders:       { label: 'Pending',             icon: '⏳', category: 'orders',    format: v => v?.toLocaleString() },
  canceled_orders:      { label: 'Canceled',            icon: '❌', category: 'orders',    format: v => v?.toLocaleString() },
  retail_orders:        { label: 'Retail Invoices',     icon: '🧾', category: 'orders',    format: v => v?.toLocaleString() },

  // Logistics
  on_time_delivery_rate:{ label: 'On-Time Delivery',   icon: '🚚', category: 'logistics', format: v => `${v?.toFixed(1)}%` },
  freight_ratio:        { label: 'Freight Ratio',       icon: '📬', category: 'logistics', format: v => `${v?.toFixed(1)}%` },

  // Customer Satisfaction
  avg_review_score:     { label: 'Avg Review Score',   icon: '⭐', category: 'satisfaction', format: v => `${v?.toFixed(2)} / 5` },
  cancellation_rate:    { label: 'Cancellation Rate',  icon: '🚫', category: 'satisfaction', format: v => `${v?.toFixed(1)}%` },

  // Payments
  avg_payment_installments: { label: 'Avg Installments', icon: '💳', category: 'payments', format: v => `${v?.toFixed(1)}x` },

  // Customers
  total_customers:      { label: 'Total Customers',      icon: '👥', category: 'customers', format: v => v?.toLocaleString() },
  retail_customers:     { label: 'Retail Customers (UK)', icon: '🇬🇧', category: 'customers', format: v => v?.toLocaleString() },

  // Products
  total_products:       { label: 'Total Products',      icon: '🏷️', category: 'products',  format: v => v?.toLocaleString() },
  total_sellers:        { label: 'Total Sellers',       icon: '🏪', category: 'sellers',    format: v => v?.toLocaleString() },
}

const COLORS = ['#6366F1','#06B6D4','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899','#14B8A6','#F97316','#64748B']

const CardTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 12px', fontSize: 12 }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.fill || p.stroke || 'var(--text-primary)', fontWeight: 600 }}>
            {p.name}: {typeof p.value === 'number' ? (p.name?.toLowerCase().includes('revenue') ? `R$${p.value.toLocaleString()}` : p.value.toLocaleString()) : p.value}
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
        <span className="kpi-icon" style={{ fontSize: 18 }}>{cfg.icon}</span>
      </div>
      {val !== null && val !== undefined ? (
        <div className="kpi-value" style={{ color: 'var(--indigo)' }}>
          {cfg.format(val)}
        </div>
      ) : (
        <div className="kpi-value" style={{ color: 'var(--text-muted)', fontSize: 16 }}>—</div>
      )}
      {data.unit && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{data.unit}</div>
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
      .then(res => {
        // FastAPI returns { status, data: { kpi_id: {...} } }
        setKpis(res.data?.data || {})
        setLast(new Date().toLocaleTimeString())
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchKPIs() }, [])

  // Chart data transforms
  const monthlyTrend = kpis?.monthly_revenue_trend?.data || []
  const catData = (kpis?.top_categories_by_revenue?.data || []).map(c => ({
    name: (c.category || 'Unknown').replace(/_/g, ' ').slice(0, 22),
    revenue: Math.round(c.revenue || 0),
    orders: c.order_count || 0,
  }))
  const geoData = (kpis?.customer_geographic_concentration?.data || []).map((g, i) => ({
    name: g.state || '?',
    revenue: Math.round(g.revenue || 0),
    fill: COLORS[i % COLORS.length],
  }))
  const paymentData = (kpis?.payment_type_distribution?.data || []).map((p, i) => ({
    name: (p.payment_type || 'Unknown').replace(/_/g, ' '),
    value: Number(p.count || 0),
    fill: COLORS[i % COLORS.length],
  }))
  const reviewData = (kpis?.review_score_distribution?.data || []).map(r => ({
    score: `Score ${r.score}`,
    count: Number(r.count || 0),
  }))
  const orderStatusData = (kpis?.order_status_breakdown?.data || []).map((s, i) => ({
    name: (s.status || 'Unknown').replace(/_/g, ' '),
    count: Number(s.count || 0),
    fill: COLORS[i % COLORS.length],
  }))
  const topProducts = (kpis?.top_selling_products?.data || []).map(p => ({
    name: (p.product || 'Unknown').replace(/_/g, ' ').slice(0, 20),
    revenue: Math.round(p.revenue || 0),
    orders: Number(p.order_count || 0),
  }))

  if (loading) return <LoadingSpinner message="Fetching live KPIs from BISFT database..." />
  if (error) return (
    <div className="fade-up">
      <PageHeader title="Business Intelligence" subtitle="BISFT · PostgreSQL Powered" />
      <div className="error-banner">⚠️ {error} — Make sure FastAPI is running on port 8000.</div>
    </div>
  )

  return (
    <div className="fade-up">
      <PageHeader title="Business Intelligence Overview" subtitle="Live KPIs from BISFT PostgreSQL Database">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {lastUpdated && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Updated {lastUpdated}</span>}
          <button className="btn btn-ghost" onClick={fetchKPIs}>🔄 Refresh</button>
        </div>
      </PageHeader>

      {/* Revenue KPIs */}
      <h3 style={{ marginTop: 8, marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>💰 Revenue</h3>
      <div className="kpi-grid">
        {['revenue', 'revenue_delivered', 'total_revenue_retail', 'aov', 'aov_delivered'].map(id => (
          <KPICard key={id} id={id} data={kpis?.[id]} />
        ))}
      </div>

      {/* Orders KPIs */}
      <h3 style={{ marginTop: 24, marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>📋 Orders</h3>
      <div className="kpi-grid">
        {['total_orders', 'delivered_orders', 'pending_orders', 'canceled_orders', 'retail_orders'].map(id => (
          <KPICard key={id} id={id} data={kpis?.[id]} />
        ))}
      </div>

      {/* Logistics & Satisfaction KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 24 }}>
        <div>
          <h3 style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>🚚 Logistics</h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            {['on_time_delivery_rate', 'freight_ratio'].map(id => <KPICard key={id} id={id} data={kpis?.[id]} />)}
          </div>
        </div>
        <div>
          <h3 style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>⭐ Satisfaction</h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            {['avg_review_score', 'cancellation_rate', 'avg_payment_installments'].map(id => <KPICard key={id} id={id} data={kpis?.[id]} />)}
          </div>
        </div>
      </div>

      {/* Customers & Products KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 24 }}>
        <div>
          <h3 style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>👥 Customers</h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            {['total_customers', 'retail_customers'].map(id => <KPICard key={id} id={id} data={kpis?.[id]} />)}
          </div>
        </div>
        <div>
          <h3 style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>🏷️ Products & Sellers</h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            {['total_products', 'total_sellers'].map(id => <KPICard key={id} id={id} data={kpis?.[id]} />)}
          </div>
        </div>
      </div>

      {/* Monthly Revenue Trend */}
      {monthlyTrend.length > 0 && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-header">
            <div>
              <p className="card-title">📈 Monthly Revenue Trend</p>
              <p className="card-subtitle">Revenue over time — Olist delivered orders</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={monthlyTrend} margin={{ left: 10, right: 20 }}>
              <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366F1" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#6366F1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
              <XAxis dataKey="month" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => `R$${(v/1000).toFixed(0)}k`} />
              <Tooltip content={<CardTooltip />} />
              <Area type="monotone" dataKey="revenue" stroke="#6366F1" fill="url(#areaGrad)" strokeWidth={2} name="Revenue" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Charts Row */}
      <div className="grid-2 section-gap">
        {/* Top Categories */}
        {catData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">🏷️ Top Categories by Revenue</p>
              <p className="card-subtitle">Olist delivered orders</p>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={catData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" horizontal={false} />
                <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => `R$${(v/1000).toFixed(0)}k`} />
                <YAxis type="category" dataKey="name" width={110} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                <Tooltip content={<CardTooltip />} />
                <Bar dataKey="revenue" name="Revenue" fill="#6366F1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Revenue by State */}
        {geoData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">🗺️ Revenue by State (Brazil)</p>
              <p className="card-subtitle">Olist delivered orders by customer location</p>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={geoData} cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={3} dataKey="revenue" nameKey="name">
                  {geoData.map((e, i) => <Cell key={i} fill={e.fill} stroke="transparent" />)}
                </Pie>
                <Tooltip formatter={(v) => [`R$${v.toLocaleString()}`, 'Revenue']} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* More Charts Row */}
      <div className="grid-2 section-gap">
        {/* Top Products */}
        {topProducts.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">🔥 Top Selling Products</p>
              <p className="card-subtitle">By revenue — Olist delivered orders</p>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topProducts} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} interval={0} angle={-20} textAnchor="end" />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={v => `R$${(v/1000).toFixed(0)}k`} />
                <Tooltip content={<CardTooltip />} />
                <Bar dataKey="revenue" name="Revenue" fill="#06B6D4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Payment Types */}
        {paymentData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">💳 Payment Types</p>
              <p className="card-subtitle">Order count by payment method — Olist</p>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={paymentData} cx="50%" cy="50%" outerRadius={85} innerRadius={40} paddingAngle={3} dataKey="value" nameKey="name">
                  {paymentData.map((e, i) => <Cell key={i} fill={e.fill} stroke="transparent" />)}
                </Pie>
                <Tooltip />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Order Status & Review Distribution */}
      <div className="grid-2 section-gap">
        {orderStatusData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">📊 Order Status Breakdown</p>
              <p className="card-subtitle">All orders by status — Olist</p>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={orderStatusData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <Tooltip content={<CardTooltip />} />
                <Bar dataKey="count" name="Orders" radius={[4, 4, 0, 0]}>
                  {orderStatusData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {reviewData.length > 0 && (
          <div className="card">
            <div className="card-header">
              <p className="card-title">⭐ Review Score Distribution</p>
              <p className="card-subtitle">Count of reviews by score — Olist</p>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={reviewData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
                <XAxis dataKey="score" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <Tooltip content={<CardTooltip />} />
                <Bar dataKey="count" name="Reviews" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* System Info */}
      <div className="card" style={{ padding: '14px 20px', marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>🗄️</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>PostgreSQL · BISFT</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>📊</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{Object.keys(kpis || {}).length} Live KPIs</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>🤖</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>LangGraph Multi-Agent</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>🔮</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Prophet Forecasting · DNN Churn</span>
          </div>
          <div style={{ marginLeft: 'auto' }}>
            <span className="badge badge-info">v3.0 · Database-First</span>
          </div>
        </div>
      </div>
    </div>
  )
}