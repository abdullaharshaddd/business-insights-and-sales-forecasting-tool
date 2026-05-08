import { NavLink, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import api from '../api/client'

const NAV_ITEMS = [
  { to: '/dashboard',   icon: '⚡', label: 'Overview' },
  { to: '/forecasting', icon: '📈', label: 'Forecasting' },
  { to: '/churn',       icon: '🎯', label: 'Churn Prediction' },
  { to: '/chat',        icon: '💬', label: 'AI Consultant' },
  { to: '/analytics',   icon: '🔬', label: 'Analytics Explorer' },
]

const OPS_ITEMS = [
  { to: '/inventory',      icon: '📦', label: 'Inventory & Products' },
  { to: '/purchase-orders',icon: '🛒', label: 'Purchase Orders' },
  { to: '/suppliers',      icon: '🏢', label: 'Suppliers' },
]

export default function Sidebar() {
  const location = useLocation()
  const [systemStatus, setSystemStatus] = useState(null)

  useEffect(() => {
    api.get('/dashboard/status')
      .then(data => setSystemStatus(data))
      .catch(() => setSystemStatus(null))
  }, [])

  const isHealthy = systemStatus?.status === 'healthy'

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-mark">
          <div className="logo-icon">🧠</div>
          <div className="logo-text">
            <span className="logo-name">BISFT</span>
            <span className="logo-tag">Decision Intelligence</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <span className="nav-section-label">Intelligence Platform</span>

        {NAV_ITEMS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}

        <span className="nav-section-label" style={{ marginTop: 16 }}>Operations (Node.js)</span>

        {OPS_ITEMS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        {systemStatus ? (
          <div className="sidebar-badge">
            <div className={`status-dot`} style={{
              background: isHealthy ? 'var(--success)' : 'var(--warning)',
              boxShadow: `0 0 6px ${isHealthy ? 'var(--success)' : 'var(--warning)'}`,
            }} />
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
              {isHealthy ? 'All systems healthy' : 'Some services degraded'}
            </span>
          </div>
        ) : (
          <div className="sidebar-badge">
            <div className="status-dot" style={{ background: 'var(--text-muted)', boxShadow: 'none' }} />
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Checking status...</span>
          </div>
        )}
        <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
          FAST-NUCES • Academic Project
        </div>
      </div>
    </aside>
  )
}
