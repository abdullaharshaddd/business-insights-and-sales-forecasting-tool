import { NavLink, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import api from '../api/client'
import { useAuth } from '../context/AuthContext'

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
]

export default function Sidebar() {
  const location = useLocation()
  const { user, logout } = useAuth()
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
        {user && (
          <div className="user-profile">
            <div className="user-avatar">{user.email[0].toUpperCase()}</div>
            <div className="user-info">
              <span className="user-email">{user.email}</span>
              <button className="logout-btn" onClick={logout}>Sign Out</button>
            </div>
          </div>
        )}

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

      <style dangerouslySetInnerHTML={{ __html: `
        .user-profile {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: var(--bg-elevated);
          border-radius: var(--radius-md);
          margin-bottom: 12px;
        }
        .user-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: var(--indigo);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 700;
          font-size: 14px;
        }
        .user-info {
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .user-email {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-primary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .logout-btn {
          background: none;
          border: none;
          color: var(--danger);
          font-size: 10px;
          font-weight: 700;
          padding: 0;
          text-align: left;
          cursor: pointer;
          opacity: 0.8;
          transition: opacity 0.2s;
        }
        .logout-btn:hover {
          opacity: 1;
          text-decoration: underline;
        }
      `}} />
    </aside>
  )
}
