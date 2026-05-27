const REASON_LABELS = {
  text_low_parser_visuals: '文本少且含视觉元素',
  text_low_visual_high: '文本少但页面视觉复杂',
  document_image_based: '疑似图片型文档',
  ocr_too_short_visual_high: 'OCR 文本不足且视觉复杂',
  garbled_text: 'OCR 文本疑似乱码',
  diagram_like: '图表/流程图需要语义理解',
  pptx_no_embedded_images: 'PPTX 当前页无可增强的嵌入图片',
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
  if (plan?.ocr_installed && plan?.ocr_cpu_available) {
    if ((plan.available_actions || []).includes('vision')) return 'GPU OCR 不可用，可选择 CPU OCR 或 Vision'
    return 'GPU OCR 不可用，可选择 CPU OCR'
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

export function formatPageLabel(page) {
  const pageNum = Number(page)
  if (Number.isFinite(pageNum)) return `第 ${pageNum + 1} 页`
  return '相关页面'
}
