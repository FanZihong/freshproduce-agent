import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'
export const USE_MOCK = String(import.meta.env.VITE_USE_MOCK ?? 'true') === 'true'

export const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json',
  },
})
