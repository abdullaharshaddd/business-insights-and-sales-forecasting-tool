import { useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

export default function Inventory() {
  const [products, setProducts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    inventoryApi.get('/products')
      .then(res => {
        console.log('Inventory Debug - Products:', res);
        const productList = Array.isArray(res?.data) ? res.data : (res?.data?.products || []);
        setProducts(productList);
        setError(null);
      })
      .catch(err => {
        console.error('Inventory Fetch Error:', err);
        setError(err.message);
      })
      .finally(() => setLoading(false))
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
                      <th>Stock Level</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {products.map(p => (
                      <tr key={p.id}>
                        <td>{p.sku}</td>
                        <td style={{ fontWeight: 500 }}>{p.name}</td>
                        <td>{p.category?.name || 'Uncategorized'}</td>
                        <td>R${p.basePrice}</td>
                        <td style={{ fontWeight: 700, textAlign: 'center' }}>
                          {p.inventory?.quantity || 0}
                        </td>
                        <td>
                          {(p.inventory?.quantity || 0) > p.reorderPoint ? (
                            <span className="badge badge-success">In Stock</span>
                          ) : (p.inventory?.quantity || 0) > 0 ? (
                            <span className="badge badge-warning">Low Stock</span>
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
