import { useState, useEffect, useCallback, useRef } from 'react'
import Sidebar from './Sidebar'
import ChatPanel from './ChatPanel'
import ReaderPanel from './ReaderPanel'
import TranslatePanel from './TranslatePanel'
import NotesPanel from './NotesPanel'
import ConfigPanel from './ConfigPanel'
import { listDocuments, uploadDocument, deleteDocument, getSections } from './api'
import './App.css'

const MIN_READER_WIDTH = 360
const MIN_WORKSPACE_WIDTH = 260
const DEFAULT_READER_WIDTH = 500

const WORKSPACE_TABS = [
  { id: 'chat', label: 'Chat' },
  { id: 'translate', label: 'Translate' },
  { id: 'study', label: 'Study Brief' },
  { id: 'config', label: 'Config' },
]

export default function App() {
  const [docs, setDocs] = useState([])
  const [active, setActive] = useState('')
  const [tab, setTab] = useState('chat')
  const [uploading, setUploading] = useState(false)
  const [sections, setSections] = useState([])
  const [activeSectionId, setActiveSectionId] = useState('')
  const [error, setError] = useState('')
  const [readerWidth, setReaderWidth] = useState(DEFAULT_READER_WIDTH)
  const dragging = useRef(false)
  const mainRef = useRef(null)

  useEffect(() => {
    listDocuments().then(data => {
      if (data.docs) setDocs(data.docs)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!active) return
    getSections(active).then(data => {
      setSections(data.sections || [])
      setActiveSectionId('')
    }).catch(() => setSections([]))
  }, [active])

  const handleUpload = useCallback(async (e) => {
    const files = e.target.files
    if (!files.length) return
    setUploading(true)
    setError('')
    let lastDocId = ''
    for (const f of files) {
      try {
        const result = await uploadDocument(f)
        if (result.doc_id) lastDocId = result.doc_id
        if (result.status === 'failed') {
          setError(result.error || `${f.name} 解析失败`)
        }
      } catch (err) {
        setError(`${f.name} 上传失败: ${err.message}`)
      }
    }
    try {
      const data = await listDocuments()
      if (data.docs) setDocs(data.docs)
      if (lastDocId) setActive(lastDocId)
    } catch (err) {
      setError(`刷新文档列表失败: ${err.message}`)
    }
    setUploading(false)
    e.target.value = ''
  }, [])

  const handleDelete = useCallback((id) => {
    deleteDocument(id).then(() => {
      setDocs(prev => prev.filter(d => d.doc_id !== id))
      if (active === id) {
        setActive('')
        setSections([])
        setActiveSectionId('')
      }
    }).catch(() => {})
  }, [active])

  // Resize
  const handleMouseDown = useCallback((e) => {
    e.preventDefault()
    dragging.current = true
    const startX = e.clientX
    const startWidth = readerWidth
    const onMove = (ev) => {
      if (!dragging.current) return
      const delta = ev.clientX - startX
      const total = mainRef.current?.getBoundingClientRect().width || window.innerWidth
      const maxReader = Math.max(MIN_READER_WIDTH, total - MIN_WORKSPACE_WIDTH - 10)
      const next = Math.min(maxReader, Math.max(MIN_READER_WIDTH, startWidth + delta))
      setReaderWidth(next)
    }
    const onUp = () => {
      dragging.current = false
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [readerWidth])

  const activeDoc = docs.find(d => d.doc_id === active) || {}
  const visibleSections = active ? sections : []
  const visibleSectionId = active ? activeSectionId : ''
  const hasDoc = !!activeDoc.doc_id

  return (
    <div className="app">
      <Sidebar docs={docs} active={active} setActive={setActive}
               onUpload={handleUpload} onDelete={handleDelete} uploading={uploading} />
      <main className="main" ref={mainRef}>
        {error && <div className="app-error">{error}</div>}
        {!hasDoc ? (
          <div className="empty-state">请先上传文档</div>
        ) : (
          <div className="reader-workspace">
            <div className="reader-pane" style={{ width: readerWidth, minWidth: readerWidth }}>
              <ReaderPanel doc={activeDoc} activeSectionId={visibleSectionId} sections={visibleSections} />
            </div>
            <div className="resize-handle" onMouseDown={handleMouseDown}>
              <div className="resize-handle-line" />
            </div>
            <div className="workspace-pane">
              <nav className="tabs">
                {WORKSPACE_TABS.map(t => (
                  <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
                    {t.label}
                  </button>
                ))}
              </nav>
              <div className="panel">
                <div className={`tab-panel ${tab === 'chat' ? 'active' : ''}`} hidden={tab !== 'chat'}>
                  <ChatPanel doc={activeDoc} activeSectionId={visibleSectionId} sections={visibleSections}
                             sectionTitle={visibleSections.find(s => s.section_id === visibleSectionId)?.title || ''} />
                </div>
                <div className={`tab-panel ${tab === 'translate' ? 'active' : ''}`} hidden={tab !== 'translate'}>
                  <TranslatePanel doc={activeDoc} sections={visibleSections} activeSectionId={visibleSectionId}
                                  onSelectSection={(s) => setActiveSectionId(s.section_id)} />
                </div>
                <div className={`tab-panel ${tab === 'study' ? 'active' : ''}`} hidden={tab !== 'study'}>
                  <NotesPanel doc={activeDoc} docId={active} />
                </div>
                <div className={`tab-panel ${tab === 'config' ? 'active' : ''}`} hidden={tab !== 'config'}>
                  <ConfigPanel />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
