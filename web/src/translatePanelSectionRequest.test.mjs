import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./TranslatePanel.jsx', import.meta.url), 'utf8')

test('section translation and explanation use the selected section id explicitly', () => {
  assert.match(source, /explicitSectionId = activeSectionId/)
  assert.match(source, /section_id: sectionId/)
  assert.match(source, /doExplain\(prompt, 'translate', cacheKey, currentSection\.section_id\)/)
  assert.doesNotMatch(source, /解释当前章节/)
  assert.doesNotMatch(source, /explainSection/)
})
