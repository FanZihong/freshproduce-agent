import { createBrowserRouter, Navigate } from 'react-router-dom'
import { AppLayout } from '@/layouts/AppLayout'
import { ChatPage } from '@/pages/chat/ChatPage'
import { OverviewPage } from '@/pages/overview/OverviewPage'
import { TicketListPage } from '@/pages/tickets/TicketListPage'
import { TicketDetailPage } from '@/pages/tickets/TicketDetailPage'
import { DatasetPage } from '@/pages/evals/DatasetPage'
import { RunPage } from '@/pages/evals/RunPage'
import { ReportPage } from '@/pages/evals/ReportPage'
import { TraceListPage } from '@/pages/traces/TraceListPage'
import { TraceDetailPage } from '@/pages/traces/TraceDetailPage'
import { KnowledgePage } from '@/pages/knowledge/KnowledgePage'
import { IntentPage } from '@/pages/intent/IntentPage'
import { RetrievalPage } from '@/pages/retrieval/RetrievalPage'
import { Empty } from '@/components/ui/Empty'

function NotFound() {
  return (
    <div className="p-6">
      <Empty title="页面不存在" description="请从左侧导航进入功能页面" />
    </div>
  )
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: 'chat', element: <ChatPage /> },
      { path: 'chat/:sessionId', element: <ChatPage /> },
      { path: 'overview', element: <OverviewPage /> },
      { path: 'tickets', element: <TicketListPage /> },
      { path: 'tickets/:ticketId', element: <TicketDetailPage /> },
      { path: 'evals', element: <DatasetPage /> },
      { path: 'evals/runs/:runId', element: <RunPage /> },
      { path: 'evals/runs/:runId/report', element: <ReportPage /> },
      { path: 'traces', element: <TraceListPage /> },
      { path: 'traces/:traceId', element: <TraceDetailPage /> },
      { path: 'retrieval', element: <RetrievalPage /> },
      { path: 'knowledge', element: <KnowledgePage /> },
      { path: 'intent', element: <IntentPage /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])
