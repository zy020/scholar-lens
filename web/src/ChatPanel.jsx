import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { streamChat } from './api'
import {
  getCitedEvidenceView,
  getEvidenceCardId,
  getEvidenceToggleLabel,
  normalizeCitationText,
} from './chatEvidence'
import 'katex/dist/katex.min.css'

function CitationMarker({ num, onClick }) {
  return <sup className="citation-marker" onClick={onClick}>[{num}]</sup>
}

function MarkdownBlock({ text }) {
  if (!text) return null
  return (
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {text}
    </ReactMarkdown>
  )
}

function numberSections(sections) {
  if (!sections || !sections.length) return []
  const counters = []
  return sections.map((section) => {
    const level = Math.max(1, Number(section.level || 1))
    counters[level - 1] = (counters[level - 1] || 0) + 1
    counters.length = level
    return {
      ...section,
      displayNumber: counters.slice(0, level).join('.'),
    }
  })
}

function cleanSectionLabel(sectionId, docId, numberedSections) {
  if (!sectionId) return ''
  if (numberedSections && numberedSections.length) {
    const sec = numberedSections.find(s => s.section_id === sectionId)
    if (sec && sec.displayNumber) {
      return `${sec.displayNumber} ${sec.title}`
    }
    if (sec) return sec.title || sectionId
  }
  const stripped = docId ? sectionId.replace(docId + '_', '') : sectionId
  return stripped || sectionId
}

export default function ChatPanel({ doc, activeSectionId, sectionTitle, sections }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [status, setStatus] = useState('')
  const [expandedEvidence, setExpandedEvidence] = useState({})
  const bottomRef = useRef(null)
  const abortRef = useRef(null)
  const bufferRef = useRef({ text: '', timer: null, assistantIdx: -1 })
  const numberedSections = useMemo(() => numberSections(sections || []), [sections])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const flushBuffer = useCallback(() => {
    const { text, timer, assistantIdx } = bufferRef.current
    if (timer) { clearTimeout(timer); bufferRef.current.timer = null }
    if (!text) return
    setMessages(prev => {
      const n = [...prev]
      if (n[assistantIdx]) n[assistantIdx] = { ...n[assistantIdx], content: text }
      return n
    })
  }, [])

  const clearChat = () => {
    abortRef.current?.abort()
    setMessages([])
    setStatus('')
    setExpandedEvidence({})
    bufferRef.current = { text: '', timer: null, assistantIdx: -1 }
  }

  const toggleEvidence = (msgIdx) => {
    setExpandedEvidence(prev => ({ ...prev, [msgIdx]: !prev[msgIdx] }))
  }

  const handleCitationClick = (idx) => {
    setExpandedEvidence(prev => {
      const next = { ...prev }
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].evidence?.length > idx) {
          next[i] = true
        }
      }
      return next
    })
    setTimeout(() => {
      const cards = document.querySelectorAll('.evidence-card')
      if (cards[idx]) cards[idx].scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }

  const send = async () => {
    const text = input.trim()
    if (!text || streaming) return
    if (doc.status !== 'ready') return
    setInput('')
    setStatus('')
    setMessages(prev => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '', evidence: [] }])
    const controller = new AbortController()
    abortRef.current = controller
    setStreaming(true)

    const assistantIdx = messages.length + 1
    bufferRef.current = { text: '', timer: null, assistantIdx }

    try {
      let full = ''
      for await (const event of streamChat(
        { message: text, doc_id: doc.doc_id || '', section_id: activeSectionId || '', mode: 'chat' },
        controller.signal
      )) {
        if (event.type === 'status') setStatus(event.message)
        else if (event.type === 'token') {
          full += event.token
          bufferRef.current.text = full
          if (!bufferRef.current.timer) {
            bufferRef.current.timer = setTimeout(() => {
              flushBuffer()
              bufferRef.current.timer = null
            }, 40)
          }
        } else if (event.type === 'evidence') {
          flushBuffer()
          setMessages(prev => {
            const n = [...prev]
            if (n[assistantIdx]) n[assistantIdx] = { ...n[assistantIdx], evidence: event.items || [] }
            return n
          })
        } else if (event.type === 'done') {
          flushBuffer()
          setMessages(prev => {
            const n = [...prev]
            if (n[assistantIdx]) n[assistantIdx] = { role: 'assistant', content: event.full || full, evidence: n[assistantIdx].evidence }
            return n
          })
        } else if (event.type === 'error') {
          flushBuffer()
          setMessages(prev => {
            const n = [...prev]
            if (n[assistantIdx]) n[assistantIdx] = { role: 'assistant', content: `错误: ${event.message}`, evidence: [] }
            return n
          })
        }
      }
    } catch (err) {
      flushBuffer()
      if (err.name !== 'AbortError') {
        setMessages(prev => {
          const n = [...prev]
          if (n[assistantIdx]) n[assistantIdx] = { role: 'assistant', content: `错误: ${err.message}`, evidence: [] }
          return n
        })
      }
    } finally {
      abortRef.current = null
      setStreaming(false)
      setStatus('')
    }
  }

  const stopChat = () => abortRef.current?.abort()

  const renderContent = (text) => {
    if (!text) return ''
    const parts = text.split(/(\[\d+\])/g)
    return parts.map((part, i) => {
      const m = part.match(/^\[(\d+)\]$/)
      if (m) {
        const num = parseInt(m[1], 10)
        return <CitationMarker key={i} num={num} onClick={() => handleCitationClick(num - 1)} />
      }
      return <span key={i} className="markdown-inline"><MarkdownBlock text={part} /></span>
    })
  }

  return (
    <div className="chat-panel">
      <div className="chat-toolbar">
        <span>{sectionTitle ? `当前章节: ${sectionTitle}` : '全文问答'}</span>
        <button onClick={clearChat} disabled={messages.length === 0 && !streaming}>清除对话</button>
      </div>
      <div className="messages">
        {messages.length === 0 && (
          <p className="empty">
            {doc.doc_id
              ? doc.status === 'ready'
                ? `已加载: ${doc.name}${sectionTitle ? ' · ' + sectionTitle : ''}。开始提问吧。`
                : `文档状态: ${doc.status}，请等待解析完成`
              : '请先上传文档'}
          </p>
        )}
        {messages.map((msg, i) => {
          const citedEvidence = getCitedEvidenceView(msg.content || '', msg.evidence || [])
          const normalizedContent = normalizeCitationText(
            msg.content || '',
            citedEvidence.map(e => e.originalIndex),
          )
          const hasEvidence = msg.evidence?.length > 0
          const isExpanded = expandedEvidence[i] || false
          const showEvidenceToggle = hasEvidence && !isExpanded
          const showEvidenceCards = hasEvidence && isExpanded

          return (
            <div key={i} className={`msg ${msg.role}`}>
              <div className="msg-content">
                {msg.content
                  ? renderContent(normalizedContent)
                  : (streaming && i === messages.length - 1 ? (status || '...') : '')}
              </div>
              {msg.role === 'assistant' && hasEvidence && (
                <div className="evidence">
                  {showEvidenceToggle && (
                    <button className="evidence-toggle" onClick={() => toggleEvidence(i)}>
                      {getEvidenceToggleLabel(citedEvidence.length, msg.evidence.length)}
                    </button>
                  )}
                  {showEvidenceCards && (
                    <>
                      <div className="evidence-header">
                        <span className="evidence-label">原文证据</span>
                        <button className="evidence-collapse-btn" onClick={() => toggleEvidence(i)}>收起</button>
                      </div>
                      {citedEvidence.length > 0 && (
                        <div className="evidence-cited">
                          {citedEvidence.map((e, j) => (
                            <div key={j}
                                 className="evidence-card"
                                 id={getEvidenceCardId(i, e.originalIndex)}>
                              <div className="evidence-meta">
                                <span className="evidence-ref">[{e.displayIndex}]</span>
                                <span className="evidence-source">Section: {cleanSectionLabel(e.section_id, doc.doc_id, numberedSections)}</span>
                                {e.page != null && <span className="evidence-page">· Page: {e.page}</span>}
                                <span className="evidence-score">Score: {(e.score || 0).toFixed(1)}</span>
                              </div>
                              <blockquote>{e.quote?.slice(0, 250)}{(e.quote?.length || 0) > 250 ? '...' : ''}</blockquote>
                            </div>
                          ))}
                        </div>
                      )}
                      {citedEvidence.length === 0 && (
                        <p className="evidence-uncited">检索到 {msg.evidence.length} 条证据，但回答未显式引用</p>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input">
        <input value={input} onChange={e => setInput(e.target.value)}
               onKeyDown={e => e.key === 'Enter' && send()}
               placeholder="输入问题..." disabled={streaming || doc.status !== 'ready'} />
        <button onClick={send} disabled={streaming || !input.trim() || doc.status !== 'ready'}>发送</button>
        {streaming && <button onClick={stopChat}>停止</button>}
      </div>
    </div>
  )
}
