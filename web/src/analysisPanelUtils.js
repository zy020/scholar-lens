export function analysisStatusLabel(analysis) {
  if (!analysis || analysis.status === 'missing') return '未生成'
  if (analysis.source === 'llm') return 'LLM 增强'
  if (analysis.source === 'parse_quality') return '解析质量'
  if (analysis.source === 'parser') return '解析结构'
  if (analysis.source === 'unavailable') return '需要配置模型'
  if (analysis.source === 'fallback') return '需要重新生成'
  return '可用'
}

export function termPreview(terms = [], limit = 12) {
  return terms.slice(0, limit).map(term => {
    const english = term.english || term.term || ''
    const chinese = term.chinese || term.explanation_zh || ''
    return chinese ? `${english}（${chinese}）` : english
  }).filter(Boolean)
}

export function sectionSummaryItems(summaries = {}, limit = 6) {
  return Object.entries(summaries)
    .slice(0, limit)
    .map(([sectionId, summary], index) => ({ sectionId, label: formatSectionSummaryLabel(sectionId, index), summary }))
}

export function formatSectionSummaryLabel(sectionId, index = 0) {
  const text = String(sectionId || '').trim()
  const slideMatch = text.match(/^slide_(\d+)$/)
  if (slideMatch) return `Slide ${Number(slideMatch[1]) + 1}`
  if (/^\d+$/.test(text)) return `章节 ${Number(text) + 1}`
  if (/^[a-z]+_[a-z0-9]{8,}/i.test(text) || /^[a-z0-9]{16,}$/i.test(text)) return `章节 ${index + 1}`
  return text || `章节 ${index + 1}`
}

export function parseQualityStatusLabel(status) {
  if (status === 'enhanced_completed') return '已完成解析增强'
  if (status === 'needs_enhancement') return '需要增强解析'
  if (status === 'usable') return '解析质量可用'
  return '解析质量未知'
}

export function parseQualityActionLabel(action) {
  if (action === 'ocr') return '低文本页'
  if (action === 'vision') return '建议 Vision'
  if (action === 'keep') return '保留当前结果'
  return action || '待判断'
}

export function parseQualityPageItems(analysis, limit = 6) {
  return (analysis?.parse_quality_pages || [])
    .slice(0, limit)
    .map((item, index) => ({
      key: `${item.page ?? index}-${item.recommended_action || 'keep'}`,
      pageLabel: item.page_label || formatSectionSummaryLabel(`page_${item.page ?? index}`, index),
      quality: item.quality || 'unknown',
      action: parseQualityActionLabel(item.recommended_action),
      score: Number.isFinite(Number(item.overall_score)) && Number(item.overall_score) > 0
        ? Number(item.overall_score).toFixed(2)
        : '',
      preview: item.text_preview || '',
    }))
}
