export function formatConceptStatus(status) {
  const labels = {
    needs_review: '需要复习',
    learning: '学习中',
    familiar: '较熟悉',
    seen: '已看过',
  }
  return labels[status] || '已记录'
}

export function formatEventType(eventType) {
  const labels = {
    chat_question: '提问',
    section_read: '阅读章节',
    explain_text: '解释选区',
    translate_text: '翻译选区',
    brief_generate: '生成简报',
    brief_view: '查看简报',
    export_obsidian: '导出笔记',
  }
  return labels[eventType] || eventType || '事件'
}

export function isReadableMemoryText(value) {
  const text = String(value || '').trim()
  if (!text) return false
  if (/^(doc_|[a-f0-9]{8,}|[A-Z0-9]{16,})$/i.test(text.replace(/[-_]/g, ''))) return false
  if (/^[a-z]+_[A-Z0-9]{12,}$/i.test(text)) return false
  if (/^[0-9a-f]{6,}[-_][0-9a-f-]{6,}$/i.test(text)) return false
  return /[A-Za-z\u4e00-\u9fff]/.test(text)
}

export function formatMemoryLocation(item = {}, currentDocId = '', currentDocName = '') {
  const sectionId = String(item.section_id || '').trim()
  const docId = String(item.doc_id || '').trim()
  const slideMatch = sectionId.match(/^slide_(\d+)$/)
  if (slideMatch) return `Slide ${Number(slideMatch[1]) + 1}`
  if (isReadableMemoryText(sectionId)) return sectionId
  if (!docId) return '全局'
  if (currentDocId && docId === currentDocId) return currentDocName || '当前文档'
  return '其他文档'
}

export function filterReadableConcepts(concepts = []) {
  return concepts.filter(item => isReadableMemoryText(item?.concept))
}

function formatCurrentPosition(position) {
  const text = String(position || '').trim()
  if (!text) return ''
  const sectionId = text.includes(':') ? text.split(':').pop() : text
  const formatted = formatMemoryLocation({ section_id: sectionId })
  if (formatted !== '全局') return formatted
  return isReadableMemoryText(text) ? text : '已记录当前位置'
}

export function summarizeMemorySnapshot(snapshot = {}) {
  const concepts = filterReadableConcepts(snapshot.concepts || [])
  const recentEvents = snapshot.recent_events || []
  const core = snapshot.core || {}
  return {
    currentPosition: formatCurrentPosition(core.current_position),
    sessionSummary: core.session_summary || '',
    eventCount: recentEvents.length,
    conceptCount: concepts.length,
    reviewCount: concepts.filter(item => item.status === 'needs_review').length,
  }
}
