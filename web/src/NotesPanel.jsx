import { useCallback, useEffect, useState } from 'react'
import {
  applyEnhancement,
  enhanceOcr,
  enhanceVision,
  evaluateParseQuality,
  getDocument,
  getDocumentAnalysis,
  getEnhancePlan,
  getStudyBrief,
} from './api'
import { analysisStatusLabel, parseQualityPageItems, parseQualityStatusLabel } from './analysisPanelUtils'
import { shouldShowEnhancePlan } from './enhancePlanUtils'
import { briefFocusLabel, briefSectionLabels, briefSourceLabel, briefTitle, formatBriefMarkdown, reviewLevelLabel, reviewQuestionsTitle, textQualityLabel } from './studyBriefUtils'

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
  const [enhancementCompleted, setEnhancementCompleted] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let ignore = false

    const loadForDoc = async () => {
      if (!docId) return

      setLoading(true)
      setError('')
      setEnhancementCompleted(false)
      try {
        const [analysisData, planData, cachedBrief] = await Promise.all([
          getDocumentAnalysis(docId),
          getEnhancePlan(docId).catch(() => null),
          getStudyBrief(docId, false).catch(() => null),
        ])
        if (!ignore) {
          setAnalysis(analysisData)
          setBrief(cachedBrief && !['not_generated', 'unavailable'].includes(cachedBrief.source) ? cachedBrief : null)
          setEnhancePlan(planData)
          setEnhancementCompleted(analysisData?.parse_quality_status === 'enhanced_completed')
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

  const refreshEnhancePlan = useCallback(async () => {
    if (!docId) return null
    const data = await getEnhancePlan(docId).catch(() => null)
    setEnhancePlan(data)
    return data
  }, [docId])

  useEffect(() => {
    if (!docId) return undefined
    const refreshCurrentPlan = () => {
      refreshEnhancePlan()
    }
    window.addEventListener('scholarlens-config-saved', refreshCurrentPlan)
    return () => window.removeEventListener('scholarlens-config-saved', refreshCurrentPlan)
  }, [docId, refreshEnhancePlan])

  const refreshDocument = async () => {
    if (!docId || !onDocumentUpdated) return
    const data = await getDocument(docId)
    onDocumentUpdated(data)
  }

  const handleAnalyze = async () => {
    if (!docId) return
    setAnalyzing(true)
    setError('')
    setAnalysisStatus('正在评估解析质量...')
    try {
      await evaluateParseQuality(docId, true)
      await loadAnalysis()
      let plan = await refreshEnhancePlan()
      let appliedAny = false
      if (shouldShowEnhancePlan(plan)) {
        const canRunOcr = (plan.available_actions || []).includes('gpu_ocr')
        const shouldRunOcr = canRunOcr && (
          Number(plan.estimated_ocr_pages || plan.recommended_ocr_pages?.length || 0) > 0 ||
          (plan.page_decisions || []).some(item => ['apply_ocr', 'apply_ocr_then_maybe_vision'].includes(item.action))
        )
        if (shouldRunOcr) {
          setAnalysisStatus('正在进行 OCR 增强...')
          const ocr = await enhanceOcr(docId, 'auto')
          if (hasUsableEnhancementText(ocr, 'ocr_quality')) {
            const applied = await applyEnhancement(docId)
            appliedAny = applied.status === 'applied'
            await refreshDocument()
            await loadAnalysis()
          }
        }
        plan = await refreshEnhancePlan()
        if (plan?.vision_possible) {
          setAnalysisStatus('正在执行 Vision 增强...')
          const vision = await enhanceVision(docId)
          if (hasUsableEnhancementText(vision, 'vision_quality')) {
            const applied = await applyEnhancement(docId)
            appliedAny = appliedAny || applied.status === 'applied'
            await refreshDocument()
            await loadAnalysis()
          }
        }
        plan = await refreshEnhancePlan()
      }
      const refreshedAnalysis = await loadAnalysis()
      setEnhancementCompleted(appliedAny || refreshedAnalysis?.parse_quality_status === 'enhanced_completed')
      setAnalysisStatus('')
    } catch (err) {
      setError(err.message)
      setAnalysisStatus('解析增强失败，请检查模型配置后重试')
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
    a.download = `${safeName}-文档学习分析.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const labels = briefSectionLabels(brief)
  const statusLabel = analysisStatusLabel(analysis)
  const parseQualityPages = parseQualityPageItems(analysis, 6)
  const hasEnhancementSuggestion = shouldShowEnhancePlan(enhancePlan)
  const analysisEnhancedCompleted = enhancementCompleted || analysis?.parse_quality_status === 'enhanced_completed'

  if (!doc.name) {
    return <div className="notes-panel"><p className="empty">请先上传文档</p></div>
  }

  return (
    <div className="notes-panel paper-brief-panel">
      <div className="brief-header">
        <div>
          <h3>{brief ? briefTitle(brief) : '文档分析'}</h3>
          <p className="doc-title"><strong>{doc.name}</strong></p>
        </div>
        <div className="brief-header-actions">
          {brief && <span className={`brief-source ${brief.source}`}>{briefSourceLabel(brief.source)}</span>}
          {brief?.text_quality && <span className={`brief-source ${brief.text_quality}`}>解析质量：{textQualityLabel(brief.text_quality)}</span>}
          {brief?.ocr_needed && <span className="brief-source recommended">存在低质量页</span>}
          <button onClick={handleAnalyze} disabled={loading || analyzing}>{analyzing ? '增强中...' : '解析增强'}</button>
          <button onClick={() => loadBrief(true)} disabled={loading}>{loading ? '生成中...' : (brief ? '重新生成' : '生成文档学习分析')}</button>
          <button onClick={handleExport} disabled={!brief}>导出 Markdown</button>
        </div>
      </div>

      {error && <p className="empty">加载失败: {error}</p>}
      {analysisStatus && <p className="brief-muted">{analysisStatus}</p>}
      {loading && !brief && <p className="empty">正在生成文档学习分析...</p>}
      {!loading && !brief && <p className="empty">文档学习分析尚未生成。点击“生成文档学习分析”后将调用已配置的 LLM。</p>}
      {brief?.error && <p className="brief-warning">{brief.error}</p>}

      <section className="analysis-panel">
        <div className="analysis-panel-header">
          <h4>文档解析质量分析</h4>
          <span className={`analysis-status ${analysis?.source || 'missing'}`}>{statusLabel}</span>
        </div>
        <div className="analysis-metrics">
          <span>{parseQualityStatusLabel(analysis?.parse_quality_status)}</span>
          {analysis?.updated_at && <span>更新: {analysis.updated_at}</span>}
        </div>
        {analysis?.error && <p className="brief-warning">{analysis.error}</p>}
        <div className="analysis-row">
          {analysisEnhancedCompleted && <p className="brief-muted">解析增强已完成，当前阅读、检索和问答将使用增强后的解析结果。</p>}
          {!analysisEnhancedCompleted && analysis?.parse_quality_message && <p className="brief-warning">{analysis.parse_quality_message}</p>}
          {!analysisEnhancedCompleted && (analysis?.parse_quality_warnings || []).map((warning, i) => (
            <p key={i} className="brief-muted">{warning}</p>
          ))}
          {!analysisEnhancedCompleted && (analysis?.parse_quality_actions || []).length > 0 && (
            <ul className="analysis-summary-list">
              {analysis.parse_quality_actions.map((action, i) => (
                <li key={i}><span>建议</span>{action}</li>
              ))}
            </ul>
          )}
          {!enhancementCompleted && !analysisEnhancedCompleted && parseQualityPages.length > 0 && (
            <div className="enhance-page-list">
              {parseQualityPages.map(item => (
                <span key={item.key} className="enhance-reason-tag">
                  {item.pageLabel}：{item.action}{item.score ? ` / ${item.score}` : ''}
                </span>
              ))}
            </div>
          )}
          {!enhancementCompleted && !analysisEnhancedCompleted && hasEnhancementSuggestion && (
            <p className="brief-muted">部分页面解析质量仍偏低。系统已按当前配置完成可用的自动增强；如仍需改善，可配置 Vision 后点击“解析增强”进行进一步分析。</p>
          )}
        </div>
      </section>

      {brief && (
        <>
          <section className="brief-section">
            <h4>文档学习分析</h4>
          </section>

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

          {brief.reading_focus?.length > 0 && (
            <section className="brief-section">
              <h4>{labels.focus}</h4>
              <ul>{brief.reading_focus.map((item, i) => <li key={i}><strong>{briefFocusLabel(item, brief)}</strong>: {item.reason}</li>)}</ul>
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
