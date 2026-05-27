export function getCitedIndexes(text, evidenceLength) {
  if (!text || !evidenceLength) return []
  const seen = []
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    const idx = Number(match[1]) - 1
    if (idx >= 0 && idx < evidenceLength && !seen.includes(idx)) {
      seen.push(idx)
    }
  }
  return seen
}

export function getCitedEvidence(text, evidence = []) {
  return getCitedIndexes(text, evidence.length).map(idx => ({
    ...evidence[idx],
    originalIndex: idx,
  }))
}

export function getCitedEvidenceView(text, evidence = []) {
  return getCitedIndexes(text, evidence.length).map((originalIndex, position) => ({
    quote: evidence[originalIndex]?.quote,
    doc_id: evidence[originalIndex]?.doc_id,
    section_id: evidence[originalIndex]?.section_id,
    page: evidence[originalIndex]?.page,
    chunk_id: evidence[originalIndex]?.chunk_id,
    score: evidence[originalIndex]?.score,
    originalIndex,
    displayIndex: position + 1,
  }))
}

export function normalizeCitationText(text, citedOriginalIndexes = []) {
  if (!text || citedOriginalIndexes.length === 0) return text || ''
  const displayByOriginal = new Map(
    citedOriginalIndexes.map((originalIndex, position) => [originalIndex, position + 1]),
  )
  return text.replace(/\[(\d+)\]/g, (match, raw) => {
    const originalIndex = Number(raw) - 1
    const displayIndex = displayByOriginal.get(originalIndex)
    return displayIndex ? `[${displayIndex}]` : match
  })
}

export function getOriginalIndexForDisplayCitation(displayIndex, citedEvidence = []) {
  const item = citedEvidence.find(e => e.displayIndex === displayIndex)
  return item ? item.originalIndex : null
}

export function getEvidenceToggleLabel(citedCount, retrievedCount) {
  if (citedCount > 0) return `原文证据 · 已引用 ${citedCount} 条`
  return `原文证据 · 检索到 ${retrievedCount} 条，回答未显式引用`
}

export function getEvidenceCardId(messageIndex, originalEvidenceIndex) {
  return `evidence-${messageIndex}-${originalEvidenceIndex}`
}
