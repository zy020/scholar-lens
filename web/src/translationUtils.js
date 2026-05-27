export function buildSectionTranslationPrompt(section, sectionText) {
  const title = section?.title || 'Untitled section'
  return [
    'Translate the following academic section into clear Chinese for a university student.',
    'Preserve model names, dataset names, method names, acronyms, formulas, and domain-specific terms in English when the Chinese translation would lose precision.',
    'For important technical terms, prefer the format "English term（中文解释）" on first mention, then keep the English term when appropriate.',
    'Only output the translated section content. Do not append related terms, glossary, notes, or explanations.',
    '',
    `Section: ${title}`,
    '',
    sectionText || '',
  ].join('\n')
}

export function buildSectionCacheKey(docId, sectionId, mode, sectionText) {
  return `${docId || ''}|${sectionId || ''}|${mode}|${stableTextHash(sectionText || '')}`
}

export function isCoursewareDocument(doc) {
  const type = doc?.doc_type || ''
  return type === 'slides_pdf' || type === 'courseware_pptx' || type === 'courseware' || type === 'lecture_slide'
}

export function buildSectionLabel(section, doc = {}) {
  if (!isCoursewareDocument(doc)) return section?.title || 'Untitled section'
  const slideMatch = String(section?.section_id || '').match(/^slide_(\d+)$/)
  if (slideMatch) return `Slide ${Number(slideMatch[1]) + 1}`
  if (section?.page_start != null) return `Slide ${Number(section.page_start) + 1}`
  return section?.title || 'Slide'
}

function stableTextHash(text) {
  let hash = 5381
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash << 5) + hash) ^ text.charCodeAt(i)
    hash >>>= 0
  }
  return `${text.length}:${hash.toString(36)}`
}
