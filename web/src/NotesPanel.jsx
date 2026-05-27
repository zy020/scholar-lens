import { useEffect, useState } from 'react'
import {
  analyzeDocument,
  applyEnhancement,
  enhanceOcr,
  enhanceVision,
  getDocument,
  getDocumentAnalysis,
  getEnhancePlan,
  getStudyBrief,
} from './api'
import { analysisStatusLabel, parseQualityPageItems, parseQualityStatusLabel, sectionSummaryItems, termPreview } from './analysisPanelUtils'
import { analyzeStatusText } from './analyzeDocumentUtils'
import { enhancePlanSummary, enhanceReasonItems, escalationLabels, ocrCapabilityLabel, visionPlanLabel } from './enhancePlanUtils'
import { briefSectionLabels, briefSourceLabel, briefTitle, formatBriefMarkdown, reviewLevelLabel, reviewQuestionsTitle, textQualityLabel } from './studyBriefUtils'

const ACTION_STATUS_LABELS = {
  completed: '已完成',
  applied: '已应用',
  skipped: '无需处理',
  failed: '未完成',
  unavailable: '不可用',
}

const ACTION_LABELS = {
  ocr: 'OCR 增强',
  vision: 'Vision 增强',
  apply: '应用增强结果',
}

function hasUsableEnhancementText(result, qualityKey) {
  return (result?.pages || []).some(page => {
    const quality = String(page?.[qualityKey] || page?.quality || '')
    return quality !== 'failed' && String(page?.text || '').trim()
  })
}

export default function NotesPanel({ doc, docId, onDocumentUpdated }) {
  const [brief, setBrief] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisStatus, setAnalysisStatus] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [enhancePlan, setEnhancePlan] = useState(null)
  const [enhancing, setEnhancing] = useState('')
  const [enhanceStatus, setEnhanceStatus] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let ignore = false

    const loadForDoc = async () => {
      if (!docId) return

      setLoading(true)
      setError('')
      try {
        const [analysisData, planData] = await Promise.all([
          getDocumentAnalysis(docId),
          getEnhancePlan(docId).catch(() => null),
        ])
        if (!ignore) {
          setAnalysis(analysisData)
          setBrief(null)
          setEnhancePlan(planData)
        }
      } catch (err) {
        if (!ignore) setError(err.message)
      } finally {
        if (!ignore) setLoading(false)
      }
    }

    loadForDoc()

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

  const loadAnalysis = async () => {
    if (!docId) return null
    const data = await getDocumentAnalysis(docId)
    setAnalysis(data)
    return data
  }

  const refreshEnhancePlan = async () => {
    if (!docId) return null
    const data = await getEnhancePlan(docId).catch(() => null)
    setEnhancePlan(data)
    return data
  }

  useEffect(() => {
    if (!docId) return undefined
    const refreshCurrentPlan = () => {
      refreshEnhancePlan()
    }
    window.addEventListener('scholarlens-config-saved', refreshCurrentPlan)
    return () => window.removeEventListener('scholarlens-config-saved', refreshCurrentPlan)
  }, [docId])

  const refreshDocument = async () => {
    if (!docId || !onDocumentUpdated) return
    const data = await getDocument(docId)
    onDocumentUpdated(data)
  }

  const handleEnhanceAction = async (action) => {
    if (!docId) return
    setEnhancing(action)
    setError('')
    setEnhanceStatus('')
    try {
      let result
      if (action === 'ocr') {
        result = await enhanceOcr(docId, 'auto')
      } else if (action === 'vision') {
        result = await enhanceVision(docId)
        if (hasUsableEnhancementText(result, 'vision_quality')) {
          const applied = await applyEnhancement(docId)
          await refreshDocument()
          await loadAnalysis()
          result = {
            ...result,
            status: applied.status === 'applied' ? 'applied' : result.status,
            pages: result.pages || [],
            message: applied.message || result.message,
          }
        }
      } else {
        result = await applyEnhancement(docId)
        await refreshDocument()
        await loadAnalysis()
      }
      await refreshEnhancePlan()
      const status = result?.status ? `（${ACTION_STATUS_LABELS[result.status] || result.status}）` : ''
      const pageCount = Array.isArray(result?.pages) ? `，页数 ${result.pages.length}` : ''
      setEnhanceStatus(`${ACTION_LABELS[action] || '增强操作'}${status}${pageCount}`)
    } catch (err) {
      setError(err.message)
      setEnhanceStatus('增强操作失败')
    } finally {
      setEnhancing('')
    }
  }

  const handleAnalyze = async () => {
    if (!docId) return
    setAnalyzing(true)
    setError('')
    setAnalysisStatus('')
    try {
      const result = await analyzeDocument(docId)
      setAnalysisStatus(analyzeStatusText(result))
      await loadAnalysis()
    } catch (err) {
      setError(err.message)
      setAnalysisStatus('分析失败，请检查模型配置后重试')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleExport = () => {
    const md = formatBriefMarkdown(brief)
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const safeName = String(doc.name || 'study-brief').replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]+/g, '_')
    a.download = `${safeName}-学习简报.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const labels = briefSectionLabels(brief)
  const statusLabel = analysisStatusLabel(analysis)
  const analysisTerms = termPreview(analysis?.key_terms || [], 10)
  const analysisSummaries = sectionSummaryItems(analysis?.l0_summaries || {}, 6)
  const parseQualityPages = parseQualityPageItems(analysis, 6)
  const enhanceReasons = enhanceReasonItems(enhancePlan)
  const enhanceEscalations = escalationLabels(enhancePlan)

  if (!doc.name) {
    return <div className="notes-panel"><p className="empty">请先上传文档</p></div>
  }

  return (
    <div className="notes-panel paper-brief-panel">
      <div className="brief-header">
        <div>
          <h3>{brief ? briefTitle(brief) : '学习简报'}</h3>
          <p className="doc-title"><strong>{doc.name}</strong></p>
        </div>
        <div className="brief-header-actions">
          {brief && <span className={`brief-source ${brief.source}`}>{briefSourceLabel(brief.source)}</span>}
          {brief?.text_quality && <span className={`brief-source ${brief.text_quality}`}>文本：{textQualityLabel(brief.text_quality)}</span>}
          {brief?.ocr_needed && <span className="brief-source recommended">建议 OCR</span>}
          <button onClick={handleAnalyze} disabled={loading || analyzing}>{analyzing ? '分析中...' : '增强分析'}</button>
          <button onClick={() => loadBrief(true)} disabled={loading}>{loading ? '生成中...' : (brief ? '重新生成' : '生成学习简报')}</button>
          <button onClick={handleExport} disabled={!brief}>导出 Markdown</button>
        </div>
      </div>

      {error && <p className="empty">加载失败: {error}</p>}
      {analysisStatus && <p className="brief-muted">{analysisStatus}</p>}
      {loading && !brief && <p className="empty">正在生成学习简报...</p>}
      {!loading && !brief && <p className="empty">学习简报尚未生成。点击“生成学习简报”后将调用已配置的 LLM。</p>}
      {brief?.error && <p className="brief-warning">{brief.error}</p>}

      <section className="analysis-panel">
        <div className="analysis-panel-header">
          <h4>文档分析</h4>
          <span className={`analysis-status ${analysis?.source || 'missing'}`}>{statusLabel}</span>
        </div>
        <div className="analysis-metrics">
          {analysis?.source === 'parse_quality' ? (
            <span>{parseQualityStatusLabel(analysis.parse_quality_status)}</span>
          ) : (
            <>
              <span>难度: {analysis?.difficulty || '未知'}</span>
              <span>预计阅读: {analysis?.estimated_reading_time ? `${analysis.estimated_reading_time} 分钟` : '未知'}</span>
            </>
          )}
          {analysis?.updated_at && <span>更新: {analysis.updated_at}</span>}
        </div>
        {analysis?.error && <p className="brief-warning">{analysis.error}</p>}
        {analysis?.source === 'parse_quality' && (
          <div className="analysis-row">
            {analysis.parse_quality_message && <p className="brief-warning">{analysis.parse_quality_message}</p>}
            {(analysis.parse_quality_warnings || []).map((warning, i) => (
              <p key={i} className="brief-muted">{warning}</p>
            ))}
            {(analysis.parse_quality_actions || []).length > 0 && (
              <ul className="analysis-summary-list">
                {analysis.parse_quality_actions.map((action, i) => (
                  <li key={i}><span>建议</span>{action}</li>
                ))}
              </ul>
            )}
            {parseQualityPages.length > 0 && (
              <div className="enhance-page-list">
                {parseQualityPages.map(item => (
                  <span key={item.key} className="enhance-reason-tag">
                    {item.pageLabel}：{item.action}{item.score ? ` / ${item.score}` : ''}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
        {enhancePlan && (
          <div className="analysis-row enhance-plan-row">
            <strong>增强解析计划</strong>
            <div className="enhance-plan-meta">
              <span>{enhancePlanSummary(enhancePlan)}</span>
              <span>{ocrCapabilityLabel(enhancePlan)}</span>
              <span>{visionPlanLabel(enhancePlan)}</span>
            </div>
            {enhancePlan.message && <p className="brief-muted">{enhancePlan.message}</p>}
            {enhanceReasons.length > 0 && (
              <div className="enhance-page-list">
                {enhanceReasons.map(item => (
                  <span key={item.page} className="enhance-reason-tag">{item.pageLabel}：{item.label}</span>
                ))}
              </div>
            )}
            {enhanceEscalations.length > 0 && (
              <div className="enhance-page-list">
                {enhanceEscalations.map(label => (
                  <span key={label} className="enhance-reason-tag muted">{label}</span>
                ))}
              </div>
            )}
            <div className="enhance-actions">
              <button onClick={() => handleEnhanceAction('ocr')} disabled={!!enhancing || enhancePlan.status === 'skipped'}>
                {enhancing === 'ocr' ? 'OCR 中...' : '执行 OCR'}
              </button>
              <button onClick={() => handleEnhanceAction('vision')} disabled={!!enhancing || !enhancePlan.vision_possible}>
                {enhancing === 'vision' ? 'Vision 增强中...' : '执行并应用 Vision'}
              </button>
            </div>
            {enhanceStatus && <p className="brief-muted">{enhanceStatus}</p>}
          </div>
        )}
        {analysisTerms.length > 0 && (
          <div className="analysis-row">
            <strong>术语</strong>
            <div className="brief-terms">
              {analysisTerms.map((term, i) => <span key={i} className="brief-term-tag">{term}</span>)}
            </div>
          </div>
        )}
        {analysisSummaries.length > 0 && (
          <div className="analysis-row">
            <strong>章节摘要</strong>
            <ul className="analysis-summary-list">
              {analysisSummaries.map(item => (
                <li key={item.sectionId}><span>{item.label}</span>{item.summary}</li>
              ))}
            </ul>
          </div>
        )}
        {analysis?.mermaid_map && (
          <details className="analysis-map">
            <summary>概念图</summary>
            <pre className="brief-mermaid">{analysis.mermaid_map}</pre>
          </details>
        )}
      </section>

      {brief && (
        <>
          <section className="brief-section">
            <h4>核心速览</h4>
            <ul>{brief.tldr.map((item, i) => <li key={i}>{item}</li>)}</ul>
          </section>

          <section className="brief-section">
            <h4>{labels.problem}</h4>
            <p>{brief.problem}</p>
            <p className="brief-muted">{brief.motivation}</p>
          </section>

          {brief.contributions?.length > 0 && (
            <section className="brief-section">
              <h4>{labels.contributions}</h4>
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

          {brief.method_walkthrough?.length > 0 && (
            <section className="brief-section">
              <h4>{labels.method}</h4>
              <ol>{brief.method_walkthrough.map((item, i) => <li key={i}><strong>{item.title}</strong><p>{item.explanation}</p></li>)}</ol>
            </section>
          )}

          {brief.key_terms?.length > 0 && (
            <section className="brief-section">
              <h4>关键术语</h4>
              <div className="brief-terms">
                {brief.key_terms.map((term, i) => (
                  <span key={i} className="brief-term-tag">{term.term}（{term.explanation_zh}）</span>
                ))}
              </div>
            </section>
          )}

          {brief.reading_focus?.length > 0 && (
            <section className="brief-section">
              <h4>{labels.focus}</h4>
              <ul>{brief.reading_focus.map((item, i) => <li key={i}><strong>{item.section_title}</strong>: {item.reason}</li>)}</ul>
            </section>
          )}

          <section className="brief-section">
            <h4>{reviewQuestionsTitle(brief)}</h4>
            <ul>{brief.review_questions.map((item, i) => <li key={i}><span className="brief-question-level">{reviewLevelLabel(item.level)}</span> {item.question}<p className="brief-muted">{item.expected_answer_hint}</p></li>)}</ul>
          </section>

          {brief.limitations?.length > 0 && (
            <section className="brief-section">
              <h4>注意事项</h4>
              <ul>{brief.limitations.map((item, i) => <li key={i}>{item}</li>)}</ul>
            </section>
          )}
        </>
      )}
    </div>
  )
}
