import assert from 'node:assert/strict'
import test from 'node:test'

import { buildSectionCacheKey, buildSectionTranslationPrompt } from './translationUtils.js'

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
