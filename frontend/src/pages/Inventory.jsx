import { useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PageHeader from '../components/PageHeader'

// ─── Helper ──────────────────────────────────────────────────────────────────

function formatCurrency(val) {
  if (val == null) return '—'
  return `R$${Number(val).toFixed(2)}`
}

function formatDate(d) {
  if (!d) return '—'
  return new Date(d).toLocaleDateString()
}

function Modal({ title, children, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="btn btn-ghost" onClick={onClose} style={{ padding: '4px 8px' }}>✕</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}

// ─── Add Stock Modal ──────────────────────────────────────────────────────────

function AddStockModal({ products, onSubmit, onClose, loading }) {
  const [productId, setProductId] = useState('')
  const [quantity, setQuantity]   = useState('')
  const [reason, setReason]       = useState('')
  const [error, setError]         = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!productId || !quantity || Number(quantity) <= 0) {
      setError('Product and quantity are required.')
      return
    }
    setError('')
    onSubmit({ productId, quantity: Number(quantity), reason })
  }

  return (
    <Modal title="Add Stock" onClose={onClose}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {error && <div style={{ color: 'var(--danger)', fontSize: 13 }}>{error}</div>}

        <label style={{ fontSize: 13, fontWeight: 600 }}>Product *</label>
        <select className="input" value={productId} onChange={e => setProductId(e.target.value)} required>
          <option value="">— Select product —</option>
          {products.map(p => (
            <option key={p.id} value={p.id}>{p.name} (SKU: {p.sku})</option>
          ))}
        </select>

        <label style={{ fontSize: 13, fontWeight: 600 }}>Quantity to Add *</label>
        <input className="input" type="number" min="1" value={quantity}
          onChange={e => setQuantity(e.target.value)} placeholder="e.g. 100" required />

        <label style={{ fontSize: 13, fontWeight: 600 }}>Reason / Note</label>
        <input className="input" value={reason} onChange={e => setReason(e.target.value)}
          placeholder="e.g. Purchase order received" />

        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="submit" className="btn btn-primary" disabled={loading} style={{ flex: 1 }}>
            {loading ? 'Adding...' : 'Add Stock'}
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClose} style={{ flex: 1 }}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  )
}

// ─── Remove Stock Modal ───────────────────────────────────────────────────────

function RemoveStockModal({ products, onSubmit, onClose, loading }) {
  const [productId, setProductId] = useState('')
  const [quantity, setQuantity]   = useState('')
  const [reason, setReason]       = useState('')
  const [error, setError]         = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!productId || !quantity || Number(quantity) <= 0) {
      setError('Product and quantity are required.')
      return
    }
    setError('')
    onSubmit({ productId, quantity: Number(quantity), reason })
  }

  return (
    <Modal title="Remove Stock" onClose={onClose}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {error && <div style={{ color: 'var(--danger)', fontSize: 13 }}>{error}</div>}

        <label style={{ fontSize: 13, fontWeight: 600 }}>Product *</label>
        <select className="input" value={productId} onChange={e => setProductId(e.target.value)} required>
          <option value="">— Select product —</option>
          {products.map(p => (
            <option key={p.id} value={p.id}>{p.name} (SKU: {p.sku})</option>
          ))}
        </select>

        <label style={{ fontSize: 13, fontWeight: 600 }}>Quantity to Remove *</label>
        <input className="input" type="number" min="1" value={quantity}
          onChange={e => setQuantity(e.target.value)} placeholder="e.g. 5" required />

        <label style={{ fontSize: 13, fontWeight: 600 }}>Reason / Note</label>
        <input className="input" value={reason} onChange={e => setReason(e.target.value)}
          placeholder="e.g. Sold, damaged, returned" />

        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="submit" className="btn btn-secondary" disabled={loading} style={{ flex: 1 }}>
            {loading ? 'Removing...' : 'Remove Stock'}
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClose} style={{ flex: 1 }}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  )
}

// ─── Adjust Stock Modal ───────────────────────────────────────────────────────

function AdjustStockModal({ products, onSubmit, onClose, loading }) {
  const [productId, setProductId] = useState('')
  const [newQty, setNewQty]       = useState('')
  const [reason, setReason]       = useState('')
  const [error, setError]         = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!productId || newQty === '' || Number(newQty) < 0) {
      setError('Product and a valid quantity are required.')
      return
    }
    setError('')
    onSubmit({ productId, newQuantity: Number(newQty), reason })
  }

  return (
    <Modal title="Adjust Stock" onClose={onClose}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {error && <div style={{ color: 'var(--danger)', fontSize: 13 }}>{error}</div>}

        <label style={{ fontSize: 13, fontWeight: 600 }}>Product *</label>
        <select className="input" value={productId} onChange={e => { setNewQty(''); setProductId(e.target.value) }} required>
          <option value="">— Select product —</option>
          {products.map(p => (
            <option key={p.id} value={p.id}>{p.name} (SKU: {p.sku})</option>
          ))}
        </select>

        <label style={{ fontSize: 13, fontWeight: 600 }}>New Quantity *</label>
        <input className="input" type="number" min="0" value={newQty}
          onChange={e => setNewQty(e.target.value)} placeholder="e.g. 250" required />

        <label style={{ fontSize: 13, fontWeight: 600 }}>Reason (required for adjustments)</label>
        <input className="input" value={reason} onChange={e => setReason(e.target.value)}
          placeholder="e.g. Stock count correction" required />

        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button type="submit" className="btn btn-warning" disabled={loading} style={{ flex: 1, color: '#fff' }}>
            {loading ? 'Adjusting...' : 'Adjust Stock'}
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClose} style={{ flex: 1 }}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  )
}

// ─── Add Product Modal ────────────────────────────────────────────────────────

function AddProductModal({ onSubmit, onClose, loading }) {
  const [form, setForm] = useState({
    name: '', sku: '', barcode: '', basePrice: '', costPrice: '',
    categoryId: '', supplierId: '', reorderPoint: '10', reorderQty: '50',
    description: '',
  })
  const [error, setError] = useState('')

  const handleChange = (field) => (e) => setForm(f => ({ ...f, [field]: e.target.value }))

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!form.name || !form.sku) { setError('Name and SKU are required.'); return }
    setError('')
    onSubmit({
      name: form.name,
      sku: form.sku,
      barcode: form.barcode || undefined,
      basePrice: form.basePrice ? Number(form.basePrice) : 0,
      costPrice: form.costPrice ? Number(form.costPrice) : undefined,
      categoryId: form.categoryId || undefined,
      supplierId: form.supplierId || undefined,
      reorderPoint: Number(form.reorderPoint) || 10,
      reorderQty: Number(form.reorderQty) || 50,
      description: form.description || undefined,
    })
  }

  return (
    <Modal title="Add New Product" onClose={onClose}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {error && <div style={{ color: 'var(--danger)', fontSize: 13 }}>{error}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Product Name *</label>
            <input className="input" value={form.name} onChange={handleChange('name')} placeholder="Widget Pro" required />
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>SKU *</label>
            <input className="input" value={form.sku} onChange={handleChange('sku')} placeholder="SKU-001" required />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Barcode (optional)</label>
            <input className="input" value={form.barcode} onChange={handleChange('barcode')} placeholder="123456789" />
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Category ID (optional)</label>
            <input className="input" value={form.categoryId} onChange={handleChange('categoryId')} placeholder="uuid" />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Base Price (R$)</label>
            <input className="input" type="number" step="0.01" value={form.basePrice} onChange={handleChange('basePrice')} placeholder="0.00" />
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Cost Price (R$)</label>
            <input className="input" type="number" step="0.01" value={form.costPrice} onChange={handleChange('costPrice')} placeholder="0.00" />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Reorder Point</label>
            <input className="input" type="number" value={form.reorderPoint} onChange={handleChange('reorderPoint')} />
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600 }}>Reorder Qty</label>
            <input className="input" type="number" value={form.reorderQty} onChange={handleChange('reorderQty')} />
          </div>
        </div>

        <div>
          <label style={{ fontSize: 12, fontWeight: 600 }}>Description</label>
          <input className="input" value={form.description} onChange={handleChange('description')} placeholder="Product description..." />
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <button type="submit" className="btn btn-primary" disabled={loading} style={{ flex: 1 }}>
            {loading ? 'Creating...' : 'Create Product'}
          </button>
          <button type="button" className="btn btn-ghost" onClick={onClose} style={{ flex: 1 }}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  )
}

// ─── Movement History Modal ───────────────────────────────────────────────────

function MovementHistoryModal({ productId, productName, onClose }) {
  const [movements, setMovements] = useState([])
  const [loading, setLoading]   = useState(true)
  const [page, setPage]        = useState(1)
  const [total, setTotal]      = useState(0)
  const limit = 20

  useEffect(() => {
    setLoading(true)
    inventoryApi.get(`/inventory/movements/${productId}`, { params: { page, limit } })
      .then(res => {
        const payload = res.data || {}
        setMovements(Array.isArray(payload.data) ? payload.data : [])
        setTotal(typeof payload.meta?.total === 'number' ? payload.meta.total : 0)
      })
      .catch(() => setMovements([]))
      .finally(() => setLoading(false))
  }, [productId, page])

  const totalPages = Math.ceil(total / limit)

  return (
    <Modal title={`Stock Movements — ${productName || productId}`} onClose={onClose}>
      {loading ? (
        <LoadingSpinner message="Loading movements..." />
      ) : movements.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)' }}>
          No stock movements recorded yet.
        </div>
      ) : (
        <>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Qty</th>
                  <th>Before → After</th>
                  <th>Reason</th>
                  <th>User</th>
                </tr>
              </thead>
              <tbody>
                {movements.map(m => (
                  <tr key={m.id}>
                    <td style={{ fontSize: 11 }}>{new Date(m.createdAt).toLocaleString()}</td>
                    <td>
                      <span className={`badge badge-${m.movementType === 'IN' ? 'success' : m.movementType === 'OUT' ? 'neutral' : 'warning'}`}>
                        {m.movementType}
                      </span>
                    </td>
                    <td style={{ fontWeight: 700, color: m.quantity > 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {m.quantity > 0 ? `+${m.quantity}` : m.quantity}
                    </td>
                    <td style={{ fontSize: 11 }}>{m.quantityBefore} → {m.quantityAfter}</td>
                    <td style={{ fontSize: 11, maxWidth: 120 }}>{m.reason || '—'}</td>
                    <td style={{ fontSize: 11 }}>{m.performedBy?.email || m.performedBy || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 12 }}>
              <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
              <span style={{ padding: '4px 12px', fontSize: 12, color: 'var(--text-muted)' }}>{page}/{totalPages}</span>
              <button className="btn btn-ghost" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next →</button>
            </div>
          )}
        </>
      )}
    </Modal>
  )
}

// ─── Main Inventory Component ─────────────────────────────────────────────────

export default function Inventory() {
  const [items, setItems]         = useState([])
  const [total, setTotal]         = useState(0)
  const [page, setPage]           = useState(1)
  const [loading, setLoading]    = useState(true)
  const [error, setError]         = useState(null)
  const [source, setSource]       = useState('all')
  const [search, setSearch]       = useState('')
  const [stats, setStats]         = useState(null)

  const [modal, setModal]         = useState(null) // 'addStock' | 'removeStock' | 'adjust' | 'addProduct' | null
  const [historyProduct, setHistoryProduct] = useState(null) // { id, name }

  const [operationalProducts, setOperationalProducts] = useState([])
  const [actionLoading, setActionLoading] = useState(false)
  const [actionMsg, setActionMsg]         = useState(null)

  const limit = 20

  // ── Fetch inventory list ────────────────────────────────────────────────────
  const fetchInventory = () => {
    setLoading(true)
    setError(null)
    inventoryApi.get('/inventory/all', { params: { page, limit, search, source } })
      .then(res => {
        const payload = res.data || {}
        setItems(Array.isArray(payload.data) ? payload.data : [])
        setTotal(typeof payload.meta?.total === 'number' ? payload.meta.total : 0)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  // ── Fetch operational products (for forms) ─────────────────────────────────
  const fetchOperationalProducts = () => {
    inventoryApi.get('/products', { params: { limit: 500, status: 'active' } })
      .then(res => {
        const list = Array.isArray(res.data) ? res.data : Array.isArray(res.data?.data) ? res.data.data : []
        setOperationalProducts(list)
      })
      .catch(() => setOperationalProducts([]))
  }

  // ── Fetch stats ────────────────────────────────────────────────────────────
  const fetchStats = () => {
    inventoryApi.get('/inventory/stats')
      .then(res => setStats(res?.data || null))
      .catch(() => {})
  }

  useEffect(() => { fetchInventory() }, [page, source])
  useEffect(() => {
    fetchStats()
    fetchOperationalProducts()
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  // ── Search ──────────────────────────────────────────────────────────────────
  const handleSearch = (e) => {
    e.preventDefault()
    setPage(1)
    fetchInventory()
  }

  const clearSearch = () => {
    setSearch('')
    setPage(1)
    fetchInventory()
  }

  // ── Stock actions ────────────────────────────────────────────────────────────
  const showMsg = (msg, type = 'success') => {
    setActionMsg({ text: msg, type })
    setTimeout(() => setActionMsg(null), 4000)
  }

  const handleAddStock = async ({ productId, quantity, reason }) => {
    setActionLoading(true)
    try {
      await inventoryApi.post('/inventory/add-stock', { productId, quantity, reason })
      showMsg(`Stock added: +${quantity} units`)
      setModal(null)
      fetchInventory()
      fetchStats()
      fetchOperationalProducts()
    } catch (e) {
      showMsg(e.message || 'Failed to add stock', 'error')
    } finally {
      setActionLoading(false)
    }
  }

  const handleRemoveStock = async ({ productId, quantity, reason }) => {
    setActionLoading(true)
    try {
      await inventoryApi.post('/inventory/remove-stock', { productId, quantity, reason })
      showMsg(`Stock removed: -${quantity} units`)
      setModal(null)
      fetchInventory()
      fetchStats()
      fetchOperationalProducts()
    } catch (e) {
      showMsg(e.message || 'Failed to remove stock', 'error')
    } finally {
      setActionLoading(false)
    }
  }

  const handleAdjustStock = async ({ productId, newQuantity, reason }) => {
    setActionLoading(true)
    try {
      await inventoryApi.post('/inventory/adjust', { productId, newQuantity, reason })
      showMsg(`Stock adjusted to ${newQuantity} units`)
      setModal(null)
      fetchInventory()
      fetchStats()
      fetchOperationalProducts()
    } catch (e) {
      showMsg(e.message || 'Failed to adjust stock', 'error')
    } finally {
      setActionLoading(false)
    }
  }

  const handleAddProduct = async (data) => {
    setActionLoading(true)
    try {
      await inventoryApi.post('/products', data)
      showMsg('Product created successfully!')
      setModal(null)
      fetchInventory()
      fetchStats()
      fetchOperationalProducts()
    } catch (e) {
      showMsg(e.message || 'Failed to create product', 'error')
    } finally {
      setActionLoading(false)
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────
  const sourceColors = { operational: 'var(--success)', retail: 'var(--indigo)', olist: 'var(--warning)' }
  const sourceLabels = { operational: 'Operational', retail: 'Online Retail (UK)', olist: 'Olist (Brazil)' }
  const totalPages = Math.ceil(total / limit)

  return (
    <div className="fade-up">
      <PageHeader
        title="Inventory & Products"
        subtitle="Complete product inventory from all data sources — synced live from PostgreSQL"
      >
        <button className="btn btn-primary" onClick={() => setModal('addProduct')}>+ Add Product</button>
      </PageHeader>

      {/* Action feedback */}
      {actionMsg && (
        <div style={{
          padding: '10px 16px', borderRadius: 8, marginBottom: 16,
          background: actionMsg.type === 'error' ? 'var(--danger)' : 'var(--success)',
          color: '#fff', fontSize: 14,
        }}>
          {actionMsg.text}
        </div>
      )}

      {/* Stats Bar */}
      {stats && (
        <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
          <div className="kpi-card" style={{ flex: 1, minWidth: 120 }}>
            <div className="kpi-card-header">
              <span className="kpi-label">Total Products</span>
              <span style={{ fontSize: 18 }}>📦</span>
            </div>
            <div className="kpi-value" style={{ color: 'var(--indigo)' }}>
              {stats.total?.toLocaleString()}
            </div>
          </div>
          <div className="kpi-card" style={{ flex: 1, minWidth: 120 }}>
            <div className="kpi-card-header">
              <span className="kpi-label">Operational</span>
              <span style={{ fontSize: 18 }}>✅</span>
            </div>
            <div className="kpi-value" style={{ color: 'var(--success)' }}>
              {stats.operational?.toLocaleString()}
            </div>
          </div>
          <div className="kpi-card" style={{ flex: 1, minWidth: 120 }}>
            <div className="kpi-card-header">
              <span className="kpi-label">Online Retail (UK)</span>
              <span style={{ fontSize: 18 }}>🇬🇧</span>
            </div>
            <div className="kpi-value" style={{ color: 'var(--indigo)' }}>
              {stats.retail?.toLocaleString()}
            </div>
          </div>
          <div className="kpi-card" style={{ flex: 1, minWidth: 120 }}>
            <div className="kpi-card-header">
              <span className="kpi-label">Olist (Brazil)</span>
              <span style={{ fontSize: 18 }}>🇧🇷</span>
            </div>
            <div className="kpi-value" style={{ color: 'var(--warning)' }}>
              {stats.olist?.toLocaleString()}
            </div>
          </div>
          <div className="kpi-card" style={{ flex: 1, minWidth: 120 }}>
            <div className="kpi-card-header">
              <span className="kpi-label">Low Stock Alerts</span>
              <span style={{ fontSize: 18 }}>⚠️</span>
            </div>
            <div className="kpi-value" style={{ color: stats.lowStockAlerts > 0 ? 'var(--danger)' : 'var(--success)' }}>
              {stats.lowStockAlerts}
            </div>
          </div>
        </div>
      )}

      {/* Stock action buttons */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <button className="btn btn-success" onClick={() => setModal('addStock')}>
          + Add Stock
        </button>
        <button className="btn btn-neutral" onClick={() => setModal('removeStock')}>
          − Remove Stock
        </button>
        <button className="btn btn-ghost" style={{ color: 'var(--warning)', border: '1px solid var(--warning)' }}
          onClick={() => setModal('adjust')}>
          ⚙ Adjust Stock
        </button>
      </div>

      {/* Filters */}
      <div className="card" style={{ marginBottom: 16, padding: '14px 20px' }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8, flex: 1 }}>
            <input
              className="input"
              placeholder="Search products..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ flex: 1, maxWidth: 400 }}
            />
            <button type="submit" className="btn btn-secondary">Search</button>
            {search && (
              <button type="button" className="btn btn-ghost" onClick={clearSearch}>
                Clear
              </button>
            )}
          </form>

          <div style={{ display: 'flex', gap: 8 }}>
            {['all', 'operational', 'retail', 'olist'].map(s => (
              <button
                key={s}
                className={`btn ${source === s ? 'btn-secondary' : 'btn-ghost'}`}
                onClick={() => { setSource(s); setPage(1) }}
              >
                {s === 'all' ? 'All' : sourceLabels[s]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="error-banner section-gap">
          ⚠️ {error} — Make sure the inventory backend is running on port 5000.
        </div>
      )}

      {/* Table */}
      {loading ? (
        <LoadingSpinner message="Loading inventory from database..." />
      ) : items.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-icon">📦</div>
            <p className="empty-title">No products found</p>
            <p className="empty-desc">Try adjusting your search or filter.</p>
          </div>
        </div>
      ) : (
        <>
          <div className="card">
            <div className="card-header">
              <p className="card-title">Product Inventory</p>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{total.toLocaleString()} total</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>ID / SKU / StockCode</th>
                    <th>Product Name</th>
                    <th>Category</th>
                    <th>Qty / Price</th>
                    <th>Supplier</th>
                    <th>Last Updated</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, idx) => (
                    <tr key={`${item.source}-${item.productId}-${idx}`}>
                      <td>
                        <span className="badge" style={{
                          background: sourceColors[item.source] + '20',
                          color: sourceColors[item.source],
                          fontSize: 10
                        }}>
                          {sourceLabels[item.source] || item.source}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'monospace', fontSize: 11 }}>
                        {item.sku || item.stockcode || item.productId?.slice(0, 12)}
                      </td>
                      <td style={{ fontWeight: 500, maxWidth: 280 }}>
                        {item.name}
                      </td>
                      <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {item.category || '—'}
                      </td>
                      <td>
                        {item.quantity !== null ? (
                          <span style={{ fontWeight: 700, color: item.quantity === 0 ? 'var(--danger)' : 'var(--text-primary)' }}>
                            {item.quantity}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>—</span>
                        )}
                        {' / '}
                        {item.latestUnitPrice !== null ? (
                          <span>{formatCurrency(item.latestUnitPrice)}</span>
                        ) : item.basePrice !== null ? (
                          <span>{formatCurrency(item.basePrice)}</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>—</span>
                        )}
                      </td>
                      <td style={{ fontSize: 12 }}>{item.supplier || '—'}</td>
                      <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {item.lastRestockAt || item.updatedAt
                          ? formatDate(item.lastRestockAt || item.updatedAt)
                          : '—'}
                      </td>
                      <td>
                        {item.source === 'operational' ? (
                          <button
                            className="btn btn-ghost"
                            style={{ padding: '3px 8px', fontSize: 11 }}
                            onClick={() => setHistoryProduct({ id: item.productId, name: item.name })}
                          >
                            History
                          </button>
                        ) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }}>
              <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                ← Prev
              </button>
              <span style={{ padding: '6px 16px', fontSize: 13, color: 'var(--text-muted)' }}>
                Page {page} of {totalPages}
              </span>
              <button className="btn btn-ghost" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                Next →
              </button>
            </div>
          )}
        </>
      )}

      {/* Modals */}
      {modal === 'addStock' && (
        <AddStockModal
          products={operationalProducts}
          onSubmit={handleAddStock}
          onClose={() => setModal(null)}
          loading={actionLoading}
        />
      )}
      {modal === 'removeStock' && (
        <RemoveStockModal
          products={operationalProducts}
          onSubmit={handleRemoveStock}
          onClose={() => setModal(null)}
          loading={actionLoading}
        />
      )}
      {modal === 'adjust' && (
        <AdjustStockModal
          products={operationalProducts}
          onSubmit={handleAdjustStock}
          onClose={() => setModal(null)}
          loading={actionLoading}
        />
      )}
      {modal === 'addProduct' && (
        <AddProductModal
          onSubmit={handleAddProduct}
          onClose={() => setModal(null)}
          loading={actionLoading}
        />
      )}
      {historyProduct && (
        <MovementHistoryModal
          productId={historyProduct.id}
          productName={historyProduct.name}
          onClose={() => setHistoryProduct(null)}
        />
      )}
    </div>
  )
}