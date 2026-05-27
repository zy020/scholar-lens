import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { streamChat } from './api'
import 'katex/dist/katex.min.css'

function MarkdownBlock({ text }) {
  if (!text) return null
  return (
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {text}
    </ReactMarkdown>
  )
}

function documentStatusLabel(status) {
  const labels = {
    uploaded: '已上传，等待解析',
    parsing: '正在解析',
    chunking: '正在切分内容',
    indexing: '正在建立索引',
    ready: '已就绪',
    failed: '解析失败',
  }
  return labels[status] || '正在处理'
}

export default function ChatPanel({ doc, activeSectionId, sectionTitle }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [status, setStatus] = useState('')
  const [deepMode, setDeepMode] = useState(false)
  const bottomRef = useRef(null)
  const abortRef = useRef(null)
  const bufferRef = useRef({ text: '', timer: null, assistantIdx: -1 })

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
    bufferRef.current = { text: '', timer: null, assistantIdx: -1 }
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
        { message: text, doc_id: doc.doc_id || '', section_id: activeSectionId || '', mode: 'chat', deep_mode: deepMode },
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

  return (
    <div className="chat-panel">
      <div className="chat-toolbar">
        <span>{sectionTitle ? `当前章节: ${sectionTitle}` : '全文问答'}</span>
        <div className="chat-toolbar-actions">
          <label className="chat-deep-toggle">
            <input type="checkbox" checked={deepMode} onChange={e => setDeepMode(e.target.checked)} disabled={streaming} />
            深度模式
          </label>
          <button onClick={clearChat} disabled={messages.length === 0 && !streaming}>清除对话</button>
        </div>
      </div>
      <div className="messages">
        {messages.length === 0 && (
          <p className="empty">
            {doc.doc_id
              ? doc.status === 'ready'
                ? `已加载: ${doc.name}${sectionTitle ? ' · ' + sectionTitle : ''}。开始提问吧。`
                : `${documentStatusLabel(doc.status)}，请稍后再提问`
              : '请先上传文档'}
          </p>
        )}
        {messages.map((msg, i) => (
            <div key={i} className={`msg ${msg.role}`}>
              <div className="msg-content">
                {msg.content
                  ? <MarkdownBlock text={msg.content} />
                  : (streaming && i === messages.length - 1 ? (status || '...') : '')}
              </div>
            </div>
        ))}
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
