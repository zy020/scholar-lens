const REASON_LABELS = {
  text_low_parser_visuals: '文本少且含视觉元素',
  text_low_visual_high: '文本少但页面视觉复杂',
  document_image_based: '疑似图片型文档',
  ocr_too_short_visual_high: 'OCR 文本不足且视觉复杂',
  garbled_text: 'OCR 文本疑似乱码',
  diagram_like: '图表/流程图需要语义理解',
  pptx_no_embedded_images: 'PPTX 当前页无可增强的嵌入图片',
  sparse_but_structured: '内容少但结构完整',
  ocr_readable_gain: 'OCR 可读文本明显提升',
  visual_semantics_need_vision: '需要理解图表/公式等视觉语义',
  ocr_first_for_visual_text: '优先提取图片文字',
  ocr_probe_failed_visual_high: 'OCR 效果差且页面视觉复杂',
  parser_recommends_ocr: '解析质量偏低',
}

const ACTION_LABELS = {
  use_original: '无需增强',
  apply_ocr: '建议 OCR',
  apply_vision: '建议 Vision',
  apply_ocr_then_maybe_vision: '先 OCR，必要时 Vision',
}

export function enhancePlanSummary(plan) {
  if (!plan || plan.status === 'skipped') return '当前无需增强解析'
  const pages = Number(plan.estimated_ocr_pages || plan.recommended_ocr_pages?.length || 0)
  return pages > 0 ? `建议增强 ${pages} 页` : '当前无需增强解析'
}

export function visionPlanLabel(plan) {
  if (!plan?.vision_available) return 'Vision 未配置'
  if (!plan?.vision_enhancement_enabled) return 'Vision 已配置，需在配置中启用增强'
  if (!plan.vision_possible) return 'Vision 已配置，当前无需调用'
  return 'Vision 可用于疑难页'
}

export function ocrCapabilityLabel(plan) {
  if (plan?.ocr_installed && plan?.ocr_gpu_available) return 'GPU OCR 可用'
  if (plan?.ocr_installed && !plan?.ocr_gpu_available) {
    if ((plan.available_actions || []).includes('vision')) return 'GPU OCR 不可用，可使用 Vision'
    return 'GPU OCR 不可用，OCR 已暂停'
  }
  if ((plan?.available_actions || []).includes('vision')) return 'RapidOCR 未安装，可使用 Vision'
  return 'RapidOCR 未安装'
}

export function enhanceReasonItems(plan, limit = 6) {
  const reasons = plan?.ocr_recommendation_reasons || {}
  return Object.entries(reasons)
    .slice(0, limit)
    .map(([page, reason]) => ({
      page,
      pageLabel: formatPageLabel(page),
      label: REASON_LABELS[reason] || reason,
    }))
}

export function escalationLabels(plan, limit = 4) {
  return (plan?.vision_escalation_reasons || [])
    .slice(0, limit)
    .map(reason => REASON_LABELS[reason] || reason)
}

export function enhanceDecisionItems(plan, limit = 8) {
  return (plan?.page_decisions || [])
    .slice(0, limit)
    .map(item => ({
      key: `${item.page}-${item.action}`,
      pageLabel: formatPageLabel(item.page),
      action: ACTION_LABELS[item.action] || item.action || '增强判断',
      reason: REASON_LABELS[item.reason] || item.reason || '',
    }))
}

export function formatPageLabel(page) {
  const pageNum = Number(page)
  if (Number.isFinite(pageNum)) return `第 ${pageNum + 1} 页`
  return '相关页面'
}
