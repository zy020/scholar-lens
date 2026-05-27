const BASE = '/api'

async function handleError(res) {
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `HTTP ${res.status}`)
  }
  return res
}

export async function getConfig() {
  const res = await fetch(`${BASE}/config`)
  return handleError(res).then(r => r.json())
}

export async function updateConfig(payload) {
  const res = await fetch(`${BASE}/config`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  })
  return handleError(res).then(r => r.json())
}

export async function listDocuments() {
  const res = await fetch(`${BASE}/documents`)
  return handleError(res).then(r => r.json())
}

async function uploadToEndpoint(file, endpoint) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}${endpoint}`, { method: 'POST', body: form })
  return handleError(res).then(r => r.json())
}

export async function uploadPaperDocument(file) {
  return uploadToEndpoint(file, '/documents/upload/paper')
}

export async function uploadCoursewareDocument(file) {
  return uploadToEndpoint(file, '/documents/upload/courseware')
}

export async function getDocument(docId) {
  const res = await fetch(`${BASE}/documents/${docId}`)
  return handleError(res).then(r => r.json())
}

export async function deleteDocument(docId) {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' })
  return handleError(res).then(r => r.json())
}

export async function getSections(docId) {
  const res = await fetch(`${BASE}/documents/${docId}/sections`)
  return handleError(res).then(r => r.json())
}

export async function getSectionText(docId, sectionId) {
  const res = await fetch(`${BASE}/documents/${docId}/sections/${encodeURIComponent(sectionId)}/text`)
  return handleError(res).then(r => r.json())
}

export async function getDocumentAnalysis(docId) {
  const res = await fetch(`${BASE}/documents/${docId}/analysis`)
  return handleError(res).then(r => r.json())
}

export async function evaluateParseQuality(docId, useLlm = false) {
  const suffix = useLlm ? '?use_llm=true' : ''
  const res = await fetch(`${BASE}/documents/${docId}/quality/evaluate${suffix}`, { method: 'POST' })
  return handleError(res).then(r => r.json())
}

export async function getEnhancePlan(docId) {
  const res = await fetch(`${BASE}/documents/${docId}/enhance-plan`, { method: 'POST' })
  return handleError(res).then(r => r.json())
}

export async function enhanceOcr(docId, mode = 'auto') {
  const suffix = mode ? `?mode=${encodeURIComponent(mode)}` : ''
  const res = await fetch(`${BASE}/documents/${docId}/enhance/ocr${suffix}`, { method: 'POST' })
  return handleError(res).then(r => r.json())
}

export async function enhanceVision(docId) {
  const res = await fetch(`${BASE}/documents/${docId}/enhance/vision`, { method: 'POST' })
  return handleError(res).then(r => r.json())
}

export async function applyEnhancement(docId) {
  const res = await fetch(`${BASE}/documents/${docId}/enhance/apply`, { method: 'POST' })
  return handleError(res).then(r => r.json())
}

export async function getStudyBrief(docId, force = false) {
  const suffix = force ? '?force=true' : ''
  const res = await fetch(`${BASE}/notes/${docId}/brief${suffix}`)
  return handleError(res).then(r => r.json())
}

export async function getNotes(docId) {
  const res = await fetch(`${BASE}/notes/${docId}`)
  return handleError(res).then(r => r.json())
}

export async function explainText(payload) {
  const { mode = 'explain', ...rest } = payload || {}
  const res = await fetch(`${BASE}/chat/explain`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, ...rest }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function* streamChat(payload, signal) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })
  await handleError(res)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() || ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6))
        } catch (err) {
          if (import.meta.env.DEV) console.warn('SSE parse failed:', line, err)
        }
      }
    }
  }
}
