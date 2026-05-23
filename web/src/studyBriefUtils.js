export function briefSourceLabel(source) {
  if (source === 'llm') return 'LLM Brief'
  if (source === 'cached') return 'Cached Brief'
  return 'Fallback Brief'
}

export function briefTitle(brief) {
  if (brief?.brief_type === 'lecture') return 'Lecture Study Brief'
  if (brief?.brief_type === 'low_text') return 'Low Text Study Brief'
  return 'Paper Understanding Brief'
}

export function formatBriefMarkdown(brief) {
  if (!brief) return ''
  const title = briefTitle(brief)
  const lines = [
    `# ${title}: ${brief.title || ''}`,
    '',
    `Source: ${briefSourceLabel(brief.source)}`,
    `Text Quality: ${brief.text_quality || 'unknown'}`,
    `OCR Needed: ${brief.ocr_needed ? 'yes' : 'no'}`,
    '',
    '## 核心速览',
    ...(brief.tldr || []).map(item => `- ${item}`),
    '',
    '## Problem & Motivation',
    brief.problem || '',
    '',
    brief.motivation || '',
    '',
    '## Contribution Map',
    ...(brief.contributions || []).map(item => {
      const evidence = item.evidence?.quote ? ` Evidence: ${item.evidence.quote}` : ''
      return `- **${item.claim}**: ${item.why_it_matters || ''}${evidence}`
    }),
    '',
    '## Method Walkthrough',
    ...(brief.method_walkthrough || []).map((item, index) => `${index + 1}. **${item.title}**: ${item.explanation}`),
    '',
    '## Key Terms',
    ...(brief.key_terms || []).map(item => `- **${item.term}**: ${item.explanation_zh || ''}`),
    '',
    '## Reading Focus',
    ...(brief.reading_focus || []).map(item => `- **${item.section_title}**: ${item.reason}`),
    '',
    '## Review Questions',
    ...(brief.review_questions || []).map(item => `- [${item.level}] ${item.question} ${item.expected_answer_hint ? `(${item.expected_answer_hint})` : ''}`),
    '',
    '## Limitations',
    ...(brief.limitations || []).map(item => `- ${item}`),
  ]
  return lines.join('\n')
}
