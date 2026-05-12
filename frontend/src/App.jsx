import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import { AuthProvider, useAuth } from './context/AuthContext'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import Churn from './pages/Churn'
import Chat from './pages/Chat'
import Analytics from './pages/Analytics'
import Inventory from './pages/Inventory'
import PurchaseOrders from './pages/PurchaseOrders'
import Login from './pages/Login'
import LoadingSpinner from './components/LoadingSpinner'

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="app-shell"><LoadingSpinner message="Checking session..." /></div>
  if (!user) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="*" element={
            <ProtectedRoute>
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
                    <Route path="/inventory"       element={<Inventory />} />
                    <Route path="/purchase-orders" element={<PurchaseOrders />} />
                  </Routes>
                </main>
              </div>
            </ProtectedRoute>
          } />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
