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
  return `${docId || ''}|${sectionId || ''}|${mode}|${sectionText?.length || 0}`
}
