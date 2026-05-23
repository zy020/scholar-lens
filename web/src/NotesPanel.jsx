import { useEffect, useState } from 'react'
import { getStudyBrief } from './api'
import { briefSourceLabel, briefTitle, formatBriefMarkdown } from './studyBriefUtils'

export default function NotesPanel({ doc, docId }) {
  const [brief, setBrief] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let ignore = false

    const loadBriefForDoc = async () => {
      if (!docId) return

      setLoading(true)
      setError('')
      try {
        const data = await getStudyBrief(docId, false)
        if (!ignore) setBrief(data)
      } catch (err) {
        if (!ignore) setError(err.message)
      } finally {
        if (!ignore) setLoading(false)
      }
    }

    loadBriefForDoc()

    return () => {
      ignore = true
    }
  }, [docId])

  const loadBrief = async (force = false) => {
    if (!docId) return
    setLoading(true)
    setError('')
    try {
      const data = await getStudyBrief(docId, force)
      setBrief(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExport = () => {
    const md = formatBriefMarkdown(brief)
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `paper-brief-${doc.doc_id || 'doc'}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!doc.name) {
    return <div className="notes-panel"><p className="empty">请先上传文档</p></div>
  }

  return (
    <div className="notes-panel paper-brief-panel">
      <div className="brief-header">
        <div>
          <h3>{brief ? briefTitle(brief) : 'Study Brief'}</h3>
          <p className="doc-title"><strong>{doc.name}</strong></p>
        </div>
        <div className="brief-header-actions">
          {brief && <span className={`brief-source ${brief.source}`}>{briefSourceLabel(brief.source)}</span>}
          {brief?.text_quality && <span className={`brief-source ${brief.text_quality}`}>Text: {brief.text_quality}</span>}
          {brief?.ocr_needed && <span className="brief-source fallback">OCR Needed</span>}
          <button onClick={() => loadBrief(true)} disabled={loading}>{loading ? '生成中...' : '重新生成'}</button>
          <button onClick={handleExport} disabled={!brief}>导出 Markdown</button>
        </div>
      </div>

      {error && <p className="empty">加载失败: {error}</p>}
      {loading && !brief && <p className="empty">正在生成 Paper Brief...</p>}
      {brief?.error && <p className="brief-warning">{brief.error}</p>}

      {brief && (
        <>
          <section className="brief-section">
            <h4>核心速览</h4>
            <ul>{brief.tldr.map((item, i) => <li key={i}>{item}</li>)}</ul>
          </section>

          <section className="brief-section">
            <h4>{brief.brief_type === 'lecture' ? '本讲主题与学习目标' : 'Problem & Motivation'}</h4>
            <p>{brief.problem}</p>
            <p className="brief-muted">{brief.motivation}</p>
          </section>

          {brief.contributions?.length > 0 && (
            <section className="brief-section">
              <h4>{brief.brief_type === 'lecture' ? '知识点脉络' : 'Contribution Map'}</h4>
              <div className="brief-card-list">
                {brief.contributions.map((item, i) => (
                  <article key={i} className="brief-card">
                    <strong>{item.claim}</strong>
                    {item.why_it_matters && <p>{item.why_it_matters}</p>}
                    {item.evidence?.quote && <blockquote>{item.evidence.section_title}: {item.evidence.quote}</blockquote>}
                  </article>
                ))}
              </div>
            </section>
          )}

          <section className="brief-section">
            <h4>{brief.brief_type === 'lecture' ? '复习路径' : 'Method Walkthrough'}</h4>
            <ol>{brief.method_walkthrough.map((item, i) => <li key={i}><strong>{item.title}</strong><p>{item.explanation}</p></li>)}</ol>
          </section>

          <section className="brief-section">
            <h4>Key Terms</h4>
            <div className="brief-terms">
              {brief.key_terms.map((term, i) => (
                <span key={i} className="brief-term-tag">{term.term}（{term.explanation_zh}）</span>
              ))}
            </div>
          </section>

          <section className="brief-section">
            <h4>{brief.brief_type === 'lecture' ? '复习重点' : 'Reading Focus'}</h4>
            <ul>{brief.reading_focus.map((item, i) => <li key={i}><strong>{item.section_title}</strong>: {item.reason}</li>)}</ul>
          </section>

          <section className="brief-section">
            <h4>Review Questions</h4>
            <ul>{brief.review_questions.map((item, i) => <li key={i}><span className="brief-question-level">{item.level}</span> {item.question}<p className="brief-muted">{item.expected_answer_hint}</p></li>)}</ul>
          </section>

          {brief.limitations?.length > 0 && (
            <section className="brief-section">
              <h4>Limitations</h4>
              <ul>{brief.limitations.map((item, i) => <li key={i}>{item}</li>)}</ul>
            </section>
          )}
        </>
      )}
    </div>
  )
}
