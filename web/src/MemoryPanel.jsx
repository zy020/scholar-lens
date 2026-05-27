import { useCallback, useEffect, useState } from 'react'
import { clearAllMemory, clearDocumentMemory, clearSessionMemory, getMemorySnapshot } from './api'
import {
  filterReadableConcepts,
  formatConceptStatus,
  formatEventType,
  formatMemoryLocation,
  summarizeMemorySnapshot,
} from './memoryPanelUtils'

export default function MemoryPanel({ docId, docName = '' }) {
  const [snapshot, setSnapshot] = useState(null)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setStatus('')
    try {
      setSnapshot(await getMemorySnapshot(docId || ''))
    } catch (err) {
      setStatus(`记忆读取失败：${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [docId])

  useEffect(() => {
    load()
  }, [load])

  const clearAndReload = useCallback(async (scope) => {
    setLoading(true)
    setStatus('')
    try {
      if (scope === 'document') {
        await clearDocumentMemory(docId)
      } else if (scope === 'all') {
        await clearAllMemory()
      } else {
        await clearSessionMemory()
      }
      setSnapshot(await getMemorySnapshot(docId || ''))
      setStatus('记忆已清理')
    } catch (err) {
      setStatus(`清理失败：${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [docId, load])

  const summary = summarizeMemorySnapshot(snapshot || {})
  const recentEvents = snapshot?.recent_events || []
  const concepts = filterReadableConcepts(snapshot?.concepts || [])

  return (
    <div className="memory-panel">
      <div className="memory-header">
        <div>
          <h3>学习记忆</h3>
          <p>用于连续学习和个性化表达，不作为事实证据来源。</p>
        </div>
        <button onClick={load} disabled={loading}>刷新</button>
      </div>

      {status && <div className="memory-status">{status}</div>}

      <div className="memory-summary">
        <div>
          <span>当前位置</span>
          <strong>{summary.currentPosition || '未记录'}</strong>
        </div>
        <div>
          <span>近期行为</span>
          <strong>{summary.eventCount}</strong>
        </div>
        <div>
          <span>概念记录</span>
          <strong>{summary.conceptCount}</strong>
        </div>
        <div>
          <span>需复习</span>
          <strong>{summary.reviewCount}</strong>
        </div>
      </div>

      {summary.sessionSummary && (
        <section className="memory-section">
          <h4>会话摘要</h4>
          <p>{summary.sessionSummary}</p>
        </section>
      )}

      <section className="memory-section">
        <h4>概念状态</h4>
        {concepts.length === 0 ? (
          <p className="memory-empty">暂无概念记录</p>
        ) : (
          <div className="memory-concepts">
            {concepts.slice(0, 16).map(item => (
              <div className="memory-concept" key={`${item.doc_id}:${item.concept}`}>
                <strong>{item.concept}</strong>
                <span>{formatConceptStatus(item.status)} · 证据 {item.evidence_count || 1} 条</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="memory-section">
        <h4>最近行为</h4>
        {recentEvents.length === 0 ? (
          <p className="memory-empty">暂无行为记录</p>
        ) : (
          <ul className="memory-events">
            {recentEvents.slice(0, 12).map(item => (
              <li key={item.id}>
                <span>{formatEventType(item.event_type)}</span>
                <em>{formatMemoryLocation(item, docId, docName)}</em>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="memory-actions">
        <button onClick={() => clearAndReload('session')} disabled={loading}>清空会话</button>
        <button onClick={() => clearAndReload('document')} disabled={loading || !docId}>清空当前文档</button>
        <button onClick={() => clearAndReload('all')} disabled={loading}>清空全部</button>
      </div>
    </div>
  )
}
