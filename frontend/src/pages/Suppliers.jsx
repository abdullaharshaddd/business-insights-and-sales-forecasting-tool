import { useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

export default function Suppliers() {
  const [suppliers, setSuppliers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    inventoryApi.get('/suppliers')
      .then(res => {
        console.log('Suppliers Debug:', res);
        const list = Array.isArray(res?.data) ? res.data : (res?.data?.suppliers || []);
        setSuppliers(list);
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="fade-up">
      <PageHeader
        title="Supplier Management"
        subtitle="Manage supplier details, contact information, and terms."
      >
        <button className="btn btn-primary" disabled={!!error}>
          + Add Supplier
        </button>
      </PageHeader>

      {error && (
        <div className="error-banner section-gap">
          ⚠️ Could not connect to Node.js Inventory Backend: {error}
        </div>
      )}

      {loading ? (
        <LoadingSpinner message="Loading suppliers from Node.js backend..." />
      ) : (
        <div className="card">
          <div className="card-header">
            <p className="card-title">All Suppliers</p>
          </div>
          
          {!error && suppliers.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Contact</th>
                    <th>Location</th>
                    <th>Lead Time</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {suppliers.map(s => (
                    <tr key={s.id}>
                      <td style={{ fontWeight: 500 }}>{s.name}</td>
                      <td>
                        <div>{s.contactPerson || 'N/A'}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{s.email || s.phone || ''}</div>
                      </td>
                      <td>{s.city ? `${s.city}, ${s.state}` : 'N/A'}</td>
                      <td>{s.leadTimeDays} days</td>
                      <td>
                        {s.isActive ? (
                          <span className="badge badge-success">Active</span>
                        ) : (
                          <span className="badge badge-neutral">Inactive</span>
                        )}
                      </td>
                      <td>
                        <button className="btn btn-ghost" style={{ padding: '4px 8px', fontSize: 12 }}>View</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🏢</div>
              <p className="empty-title">No suppliers found</p>
              <p className="empty-desc">Make sure the Node.js backend is running and the seed script has been executed.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
