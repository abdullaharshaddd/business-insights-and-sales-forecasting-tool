import { createContext, useContext, useState, useEffect } from 'react'
import { inventoryApi } from '../api/client'

const AuthContext = createContext()

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      inventoryApi.get('/auth/me')
        .then(res => {
          // sendSuccess wraps data as { success: true, data: <user> }
          setUser(res.data)
        })
        .catch(() => {
          localStorage.removeItem('token')
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  const login = async (email, password) => {
    // sendSuccess wraps result as { success: true, data: { accessToken, refreshToken, user } }
    const res = await inventoryApi.post('/auth/login', { email, password })
    const { accessToken, user: userData } = res.data
    localStorage.setItem('token', accessToken)
    setUser(userData)
    return userData
  }

  const logout = () => {
    localStorage.removeItem('token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
