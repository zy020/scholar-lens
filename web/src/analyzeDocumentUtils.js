export function analyzeStatusText(result) {
  if (!result) return ''
  if (result.source === 'llm' && result.status === 'enhanced') return '增强分析完成'
  if (result.source === 'unavailable') return result.error || '需要配置可用 LLM 后生成分析'
  if (result.error) return '分析失败，请检查模型配置后重试'
  if (result.source === 'parser') return '已生成解析结构'
  if (result.source === 'fallback') return '需要配置可用 LLM 后重新生成分析'
  return '分析已更新'
}
