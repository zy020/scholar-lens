import assert from 'node:assert/strict'
import test from 'node:test'

import { buildCoursewareTranslateHint, buildSectionCacheKey, buildSectionLabel, buildSectionTranslationPrompt, isCoursewareDocument } from './translationUtils.js'

test('section translation prompt uses full section text and preserves technical terms', () => {
  const text = 'Transformer-XL uses recurrence.\n'.repeat(40)
  const prompt = buildSectionTranslationPrompt({ title: 'Method' }, text)

  assert.match(prompt, /Transformer-XL uses recurrence/)
  assert.match(prompt, /Preserve model names/)
  assert.match(prompt, /acronyms/)
  assert.match(prompt, /domain-specific terms in English/)
  assert.match(prompt, /Do not append related terms/)
})

test('section cache key changes when full section text changes', () => {
  assert.notEqual(
    buildSectionCacheKey('doc', 'sec', 'translate', 'short'),
    buildSectionCacheKey('doc', 'sec', 'translate', 'a much longer section text'),
  )
})

test('section cache key changes for same-length content changes', () => {
  assert.notEqual(
    buildSectionCacheKey('doc', 'sec', 'translate', 'alpha beta'),
    buildSectionCacheKey('doc', 'sec', 'translate', 'gamma zeta'),
  )
})

test('courseware sections use slide labels instead of paper numbering', () => {
  const doc = { doc_type: 'slides_pdf' }

  assert.equal(isCoursewareDocument(doc), true)
  assert.equal(buildSectionLabel({ section_id: 'slide_2', page_start: 2, title: 'Outline' }, doc), 'Slide 3')
  assert.equal(buildSectionLabel({ section_id: 'intro', page_start: 0, title: 'Introduction' }, { doc_type: 'research_paper' }), 'Introduction')
})

test('courseware translation hint avoids low-value header footer wording', () => {
  const hint = buildCoursewareTranslateHint()

  assert.match(hint, /请在阅读区选中需要翻译的内容/)
  assert.doesNotMatch(hint, /页眉|页脚|装饰性文本/)
})
