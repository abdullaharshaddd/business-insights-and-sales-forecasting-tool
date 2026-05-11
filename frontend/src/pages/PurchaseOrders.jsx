import { useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

const STATUS_BADGE = {
  draft: 'badge-neutral',
  submitted: 'badge-info',
  confirmed: 'badge-primary',
  partial: 'badge-warning',
  received: 'badge-success',
  cancelled: 'badge-danger',
}

export default function PurchaseOrders() {
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    inventoryApi.get('/purchase-orders')
      .then(res => {
        console.log('Purchase Orders Debug:', res);
        const list = Array.isArray(res?.data) ? res.data : (res?.data?.orders || []);
        setOrders(list);
        setError(null);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="fade-up">
      <PageHeader
        title="Purchase Orders"
        subtitle="Manage inbound stock, track expected deliveries, and receive inventory."
      >
        <button className="btn btn-primary" disabled={!!error}>
          + Create PO
        </button>
      </PageHeader>

      {error && (
        <div className="error-banner section-gap">
          ⚠️ Could not connect to Node.js Inventory Backend: {error}
        </div>
      )}

      {loading ? (
        <LoadingSpinner message="Loading purchase orders from Node.js backend..." />
      ) : (
        <div className="card">
          <div className="card-header">
            <p className="card-title">Recent Purchase Orders</p>
          </div>
          
          {!error && orders.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>PO Number</th>
                    <th>Supplier</th>
                    <th>Order Date</th>
                    <th>Expected</th>
                    <th>Amount</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.map(po => (
                    <tr key={po.id}>
                      <td style={{ fontWeight: 600, color: 'var(--indigo-light)' }}>{po.poNumber}</td>
                      <td>{po.supplier?.name || 'Unknown'}</td>
                      <td>{new Date(po.orderDate).toLocaleDateString()}</td>
                      <td>{po.expectedDate ? new Date(po.expectedDate).toLocaleDateString() : 'N/A'}</td>
                      <td>{po.totalAmount ? `R$${po.totalAmount}` : '-'}</td>
                      <td>
                        <span className={`badge ${STATUS_BADGE[po.status] || 'badge-neutral'}`}>
                          {po.status}
                        </span>
                      </td>
                      <td>
                        <button className="btn btn-ghost" style={{ padding: '4px 8px', fontSize: 12 }}>
                          {po.status === 'confirmed' ? 'Receive' : 'View'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🛒</div>
              <p className="empty-title">No purchase orders found</p>
              <p className="empty-desc">Create your first purchase order to restock inventory.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
