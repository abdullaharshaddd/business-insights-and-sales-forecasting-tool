import axios from 'axios'

// Analytics & AI Backend (Python)
const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 min for AI consultant calls
  headers: { 'Content-Type': 'application/json' },
})

// Operational Inventory Backend (Node.js)
export const inventoryApi = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

const interceptors = {
  request: [(config) => config, (error) => Promise.reject(error)],
  response: [
    (response) => response.data,
    (error) => {
      const msg = error.response?.data?.detail || error.response?.data?.error?.message || error.message || 'Request failed'
      return Promise.reject(new Error(msg))
    }
  ]
}

api.interceptors.request.use(...interceptors.request)
api.interceptors.response.use(...interceptors.response)

inventoryApi.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

inventoryApi.interceptors.response.use(...interceptors.response)

export default api
