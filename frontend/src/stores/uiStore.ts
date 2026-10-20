import { create } from 'zustand'

export type DrawerTab = 'state' | 'tools' | 'retrieval'

export interface ToastItem {
  id: string
  title: string
  description?: string
  variant: 'success' | 'danger' | 'warning' | 'info'
}

interface UiState {
  drawerOpen: boolean
  drawerTab: DrawerTab
  toasts: ToastItem[]
  openDrawer: (tab?: DrawerTab) => void
  closeDrawer: () => void
  pushToast: (toast: Omit<ToastItem, 'id'>) => void
  removeToast: (id: string) => void
}

export const useUiStore = create<UiState>((set, get) => ({
  drawerOpen: false,
  drawerTab: 'state',
  toasts: [],

  openDrawer: (tab = 'state') => set({ drawerOpen: true, drawerTab: tab }),
  closeDrawer: () => set({ drawerOpen: false }),

  pushToast: (toast) => {
    const id = `toast-${Date.now()}-${Math.floor(Math.random() * 1000)}`
    set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }))
    setTimeout(() => get().removeToast(id), 3000)
  },

  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}))
