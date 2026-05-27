export function briefSourceLabel(source) {
  if (source === 'llm') return '模型生成'
  if (source === 'cached') return '已缓存'
  if (source === 'not_generated') return '尚未生成'
  if (source === 'unavailable') return '需要配置模型'
  return '未知来源'
}

export function briefTitle(brief) {
  if (brief?.brief_type === 'lecture') return '课件学习分析'
  return '论文学习分析'
}

export function textQualityLabel(quality) {
  const labels = {
    good: '良好',
    weak: '偏弱',
    image_based: '图片型',
    unknown: '未知',
  }
  return labels[quality] || quality || '未知'
}

export function reviewLevelLabel(level) {
  const labels = {
    basic: '基础',
    deep: '深入',
    critical: '批判',
  }
  return labels[level] || '复习'
}

export function briefSectionLabels(brief) {
  if (brief?.brief_type === 'lecture') {
    return {
      problem: '本讲主题与学习目标',
      contributions: '知识点脉络',
      method: '复习路径',
      focus: '重点 Slide',
    }
  }
  return {
    problem: '研究问题与动机',
    contributions: '核心贡献',
    method: '方法脉络',
    focus: '阅读重点',
  }
}

export function reviewQuestionsTitle(brief) {
  if (brief?.brief_type === 'lecture') return '自测问题'
  return '复习问题'
}

export function briefFocusLabel(item = {}, brief = {}) {
  const title = item.section_title || item.title || item.section_id || '重点内容'
  if (brief?.brief_type !== 'lecture') return title

  const slideMatch = String(item.section_id || '').match(/^slide_(\d+)$/)
  if (slideMatch) {
    const slideNo = Number(slideMatch[1]) + 1
    return title && title !== `Slide ${slideNo}` ? `Slide ${slideNo}：${title}` : `Slide ${slideNo}`
  }
  if (/^\d+$/.test(String(item.section_id || ''))) {
    const slideNo = Number(item.section_id) + 1
    return title && title !== `Slide ${slideNo}` ? `Slide ${slideNo}：${title}` : `Slide ${slideNo}`
  }
  const titleMatch = String(title).match(/^Slide\s+(\d+)/i)
  if (titleMatch) return title
  return title
}

export function formatBriefMarkdown(brief) {
  if (!brief) return ''
  const title = briefTitle(brief)
  const labels = briefSectionLabels(brief)
  const lines = [
    `# ${title}: ${brief.title || ''}`,
    '',
    `来源：${briefSourceLabel(brief.source)}`,
    `文本质量：${textQualityLabel(brief.text_quality)}`,
    `存在低质量页：${brief.ocr_needed ? '是' : '否'}`,
    '',
    '## 核心速览',
    ...(brief.tldr || []).map(item => `- ${item}`),
    '',
    `## ${labels.problem}`,
    brief.problem || '',
    '',
    brief.motivation || '',
    '',
    ...((brief.contributions || []).length > 0 ? [
      `## ${labels.contributions}`,
      ...(brief.contributions || []).map(item => {
        const evidence = item.evidence?.quote ? ` Evidence: ${item.evidence.quote}` : ''
        return `- **${item.claim}**: ${item.why_it_matters || ''}${evidence}`
      }),
      '',
    ] : []),
    ...((brief.method_walkthrough || []).length > 0 ? [
      `## ${labels.method}`,
      ...(brief.method_walkthrough || []).map((item, index) => `${index + 1}. **${item.title}**: ${item.explanation}`),
      '',
    ] : []),
    ...((brief.reading_focus || []).length > 0 ? [
      `## ${labels.focus}`,
      ...(brief.reading_focus || []).map(item => `- **${briefFocusLabel(item, brief)}**: ${item.reason}`),
      '',
    ] : []),
    `## ${reviewQuestionsTitle(brief)}`,
    ...(brief.review_questions || []).map(item => `- [${reviewLevelLabel(item.level)}] ${item.question} ${item.expected_answer_hint ? `(${item.expected_answer_hint})` : ''}`),
    '',
    '## 注意事项',
    ...(brief.limitations || []).map(item => `- ${item}`),
  ]
  return lines.join('\n')
}
