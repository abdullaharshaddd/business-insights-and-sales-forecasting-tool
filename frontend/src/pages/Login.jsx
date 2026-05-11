import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const [email, setEmail] = useState('admin@bisft.com')
  const [password, setPassword] = useState('Admin@123')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err.message || 'Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-container fade-up">
        <div className="login-header">
          <div className="login-logo">🚀</div>
          <h1 className="login-title">BISFT</h1>
          <p className="login-subtitle">Strategic Business Intelligence & Inventory Management</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          {error && <div className="error-banner" style={{ marginBottom: 20 }}>{error}</div>}
          
          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              className="form-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="name@company.com"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              className="form-input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
            />
          </div>

          <button className="btn btn-primary" style={{ width: '100%', marginTop: 10 }} disabled={loading}>
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          <p>Deterministic KPI Engine · LangGraph AI · PostgreSQL Unified Store</p>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .login-page {
          height: 100vh;
          width: 100vw;
          display: flex;
          align-items: center;
          justify-content: center;
          background: radial-gradient(circle at top left, rgba(99, 102, 241, 0.1), transparent),
                      radial-gradient(circle at bottom right, rgba(6, 182, 212, 0.1), transparent),
                      var(--bg-main);
        }
        .login-container {
          width: 100%;
          max-width: 420px;
          padding: 40px;
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        .login-header {
          text-align: center;
          margin-bottom: 32px;
        }
        .login-logo {
          font-size: 40px;
          margin-bottom: 16px;
        }
        .login-title {
          font-size: 28px;
          font-weight: 800;
          letter-spacing: -0.025em;
          margin-bottom: 8px;
          background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .login-subtitle {
          color: var(--text-muted);
          font-size: 14px;
        }
        .form-group {
          margin-bottom: 20px;
        }
        .form-label {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-secondary);
          margin-bottom: 8px;
        }
        .form-input {
          width: 100%;
          padding: 12px 16px;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          transition: all 0.2s;
        }
        .form-input:focus {
          outline: none;
          border-color: var(--indigo);
          box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        .login-footer {
          margin-top: 32px;
          text-align: center;
          font-size: 11px;
          color: var(--text-muted);
        }
      `}} />
    </div>
  )
}
