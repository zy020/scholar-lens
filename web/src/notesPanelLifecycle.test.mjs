import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

const source = readFileSync(new URL('./NotesPanel.jsx', import.meta.url), 'utf8')

test('NotesPanel loads study brief from an effect instead of render state', () => {
  assert.match(source, /import\s+\{\s*useEffect,\s*useState\s*\}\s+from\s+['"]react['"]/)
  assert.match(source, /useEffect\s*\(/)
  assert.doesNotMatch(source, /if\s*\(\s*!loaded\s*&&\s*docId\s*\)/)
  assert.doesNotMatch(source, /setLoaded\s*\(/)
})

test('NotesPanel avoids rendering empty optional brief sections', () => {
  assert.match(source, /brief\.method_walkthrough\?\.length > 0/)
  assert.match(source, /brief\.key_terms\?\.length > 0/)
  assert.match(source, /brief\.reading_focus\?\.length > 0/)
})

test('NotesPanel does not generate study brief on initial document load', () => {
  assert.doesNotMatch(source, /getStudyBrief\(docId,\s*false\)/)
  assert.match(source, /生成学习简报/)
})

test('NotesPanel refreshes current enhance plan after config save without running enhancement', () => {
  assert.match(source, /scholarlens-config-saved/)
  assert.match(source, /refreshEnhancePlan\(\)/)
  const listenerBlock = source.match(/window\.addEventListener\('scholarlens-config-saved'[\s\S]*?window\.removeEventListener\('scholarlens-config-saved'[\s\S]*?\)/)?.[0] || ''
  assert.doesNotMatch(listenerBlock, /enhanceVision|enhanceOcr|applyEnhancement/)
})

test('NotesPanel combines Vision enhancement and apply into one user action', () => {
  assert.match(source, /执行并应用 Vision/)
  assert.match(source, /hasUsableEnhancementText\(result,\s*'vision_quality'\)/)
  assert.match(source, /const applied = await applyEnhancement\(docId\)/)
  assert.doesNotMatch(source, />应用增强结果</)
})
