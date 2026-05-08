import { useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

export default function Inventory() {
  const [products, setProducts] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    // Fetch products and alerts in parallel
    Promise.all([
      inventoryApi.get('/products').catch(err => ({ error: err.message })),
      inventoryApi.get('/inventory/alerts').catch(err => ({ error: err.message }))
    ]).then(([productsRes, alertsRes]) => {
      if (productsRes.error || alertsRes.error) {
        setError(productsRes.error || alertsRes.error)
      } else {
        setProducts(productsRes.data || [])
        setAlerts(alertsRes.data || [])
      }
    }).finally(() => setLoading(false))
  }, [])

  return (
    <div className="fade-up">
      <PageHeader
        title="Inventory & Products"
        subtitle="Manage product catalog, view stock levels, and monitor low stock alerts."
      >
        <button className="btn btn-primary" disabled={!!error}>
          + Add Product
        </button>
      </PageHeader>

      {error && (
        <div className="error-banner section-gap">
          ⚠️ Could not connect to Node.js Inventory Backend: {error}
        </div>
      )}

      {loading ? (
        <LoadingSpinner message="Loading inventory data from Node.js backend..." />
      ) : (
        <>
          {/* Alerts Section */}
          {!error && alerts.length > 0 && (
            <div className="card section-gap" style={{ borderColor: 'var(--danger)', background: 'rgba(239,68,68,0.05)' }}>
              <div className="card-header">
                <p className="card-title" style={{ color: 'var(--danger)' }}>Low Stock Alerts ({alerts.length})</p>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {alerts.map((alert, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                    <span>{alert.product_name}</span>
                    <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{alert.available_qty} left (Reorder at {alert.reorder_point})</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Products Table */}
          <div className="card">
            <div className="card-header">
              <p className="card-title">Product Catalog</p>
            </div>
            
            {!error && products.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>SKU</th>
                      <th>Product Name</th>
                      <th>Category</th>
                      <th>Price</th>
                      <th>Stock Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {products.map(p => (
                      <tr key={p.id}>
                        <td>{p.sku}</td>
                        <td style={{ fontWeight: 500 }}>{p.name}</td>
                        <td>{p.category?.name || 'Uncategorized'}</td>
                        <td>R${p.base_price}</td>
                        <td>
                          {p.inventory?.available_qty > p.reorder_point ? (
                            <span className="badge badge-success">In Stock ({p.inventory.available_qty})</span>
                          ) : p.inventory?.available_qty > 0 ? (
                            <span className="badge badge-warning">Low Stock ({p.inventory.available_qty})</span>
                          ) : (
                            <span className="badge badge-danger">Out of Stock</span>
                          )}
                        </td>
                        <td>
                          <button className="btn btn-ghost" style={{ padding: '4px 8px', fontSize: 12 }}>Edit</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">📦</div>
                <p className="empty-title">No products found</p>
                <p className="empty-desc">Make sure the Node.js backend is running and the seed script has been executed.</p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
