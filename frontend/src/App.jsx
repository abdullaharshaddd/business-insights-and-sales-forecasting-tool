import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import Churn from './pages/Churn'
import Chat from './pages/Chat'
import Analytics from './pages/Analytics'
import Inventory from './pages/Inventory'
import Suppliers from './pages/Suppliers'
import PurchaseOrders from './pages/PurchaseOrders'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard"       element={<Dashboard />} />
            <Route path="/forecasting"     element={<Forecasting />} />
            <Route path="/churn"           element={<Churn />} />
            <Route path="/chat"            element={<Chat />} />
            <Route path="/analytics"       element={<Analytics />} />
            
            {/* Operational Routes */}
            <Route path="/inventory"       element={<Inventory />} />
            <Route path="/suppliers"       element={<Suppliers />} />
            <Route path="/purchase-orders" element={<PurchaseOrders />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
