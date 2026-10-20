import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import '@/styles/global.css'

const container = document.getElementById('root')

if (!container) {
  throw new Error('未找到根节点 #root')
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
