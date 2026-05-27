export function shouldShowEnhancePlan(plan) {
  if (!plan || plan.status === 'skipped') return false
  const recommendedPages = Number(plan.estimated_ocr_pages || plan.recommended_ocr_pages?.length || 0)
  const actionable = new Set(['apply_ocr', 'apply_vision', 'apply_ocr_then_maybe_vision'])
  const hasActionableDecision = (plan.page_decisions || []).some(item => actionable.has(item?.action))
  return recommendedPages > 0 || hasActionableDecision || Boolean(plan.vision_possible)
}
