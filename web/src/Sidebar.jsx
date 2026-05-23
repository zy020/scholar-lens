import { useState, useEffect } from 'react'

export default function Sidebar({ docs, active, setActive, onUpload, onDelete, uploading }) {
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')

  useEffect(() => {
    document.body.className = dark ? 'dark' : ''
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <aside className="sidebar">
      <div className="logo">ScholarLens</div>

      <label className="upload-btn">
        {uploading ? '上传中...' : '上传 PDF'}
        <input type="file" accept=".pdf" multiple onChange={onUpload} hidden />
      </label>

      <div className="doc-list">
        <h3>文档列表</h3>
        {(!docs || docs.length === 0) && <p className="empty">暂无文档</p>}
        {(docs || []).map(doc => (
          <div key={doc.doc_id} className={`doc-item ${active === doc.doc_id ? 'active' : ''}`}>
            <button className="doc-name" onClick={() => setActive(doc.doc_id)}>
              {active === doc.doc_id ? '▸ ' : ''}{doc.name}
              <span className="status-dot" style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: doc.status === 'ready' ? '#4a9' : doc.status === 'failed' ? '#c44' : '#ca8',
                marginLeft: 6,
              }} />
            </button>
            <button className="doc-del" onClick={() => onDelete(doc.doc_id)} title="删除">x</button>
          </div>
        ))}
      </div>

      <div style={{ flex: 1 }} />

      <button className="theme-toggle" onClick={() => setDark(!dark)}>
        {dark ? '☀ 日间模式' : '☾ 暗夜模式'}
      </button>
    </aside>
  )
}
