import type { Ticket } from '@/types'
import { http, USE_MOCK } from './client'
import { delay, mockTickets } from './mock'

export async function fetchTickets(): Promise<Ticket[]> {
  if (USE_MOCK) {
    await delay(200)
    return mockTickets
  }
  const { data } = await http.get<Ticket[]>('/tickets')
  return data
}

export async function fetchTicket(id: string): Promise<Ticket | undefined> {
  if (USE_MOCK) {
    await delay(160)
    return mockTickets.find((ticket) => ticket.id === id)
  }
  const { data } = await http.get<Ticket>(`/tickets/${id}`)
  return data
}

export async function createTicket(payload: {
  sessionId: string
  orderId: string
  reason: string
  summary: string
  priority: Ticket['priority']
}): Promise<Ticket> {
  if (USE_MOCK) {
    await delay(320)
    return {
      id: `TK-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}-${Math.floor(
        Math.random() * 9000 + 1000,
      )}`,
      sessionId: payload.sessionId,
      orderId: payload.orderId,
      reason: payload.reason,
      summary: payload.summary,
      status: 'pending',
      priority: payload.priority,
      createdAt: new Date().toISOString(),
      assignee: '待分配',
    }
  }
  const { data } = await http.post<Ticket>('/tickets', payload)
  return data
}

export async function updateTicketStatus(id: string, status: Ticket['status']): Promise<void> {
  if (USE_MOCK) {
    await delay(180)
    return
  }
  await http.patch(`/tickets/${id}`, { status })
}
