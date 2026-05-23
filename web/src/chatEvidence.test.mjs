import assert from 'node:assert/strict'
import test from 'node:test'

import {
  getCitedEvidence,
  getCitedEvidenceView,
  getEvidenceCardId,
  getEvidenceToggleLabel,
  normalizeCitationText,
} from './chatEvidence.js'

test('filters evidence to explicitly cited items and preserves original indexes', () => {
  const evidence = [
    { quote: 'first' },
    { quote: 'second' },
    { quote: 'third' },
  ]

  assert.deepEqual(
    getCitedEvidence('Uses [3] before [1], then repeats [3].', evidence),
    [
      { quote: 'third', originalIndex: 2 },
      { quote: 'first', originalIndex: 0 },
    ],
  )
})

test('evidence toggle label distinguishes retrieved from cited evidence', () => {
  assert.equal(getEvidenceToggleLabel(2, 5), '原文证据 · 已引用 2 条')
  assert.equal(getEvidenceToggleLabel(0, 5), '原文证据 · 检索到 5 条，回答未显式引用')
})

test('cited evidence is displayed sequentially in answer citation order', () => {
  const evidence = [
    { quote: 'first evidence' },
    { quote: 'second evidence' },
    { quote: 'third evidence' },
  ]

  assert.deepEqual(
    getCitedEvidenceView('Uses [3] first, then [1].', evidence).map(item => ({
      displayIndex: item.displayIndex,
      originalIndex: item.originalIndex,
      quote: item.quote,
    })),
    [
      { displayIndex: 1, originalIndex: 2, quote: 'third evidence' },
      { displayIndex: 2, originalIndex: 0, quote: 'first evidence' },
    ],
  )
})

test('answer citation text is normalized to sequential labels', () => {
  assert.equal(
    normalizeCitationText('Uses [3] first, then [1].', [2, 0]),
    'Uses [1] first, then [2].',
  )
})

test('evidence card ids include message and original evidence indexes', () => {
  assert.equal(getEvidenceCardId(4, 2), 'evidence-4-2')
})
