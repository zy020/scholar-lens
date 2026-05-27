import test from 'node:test'
import assert from 'node:assert/strict'
import {
  filterReadableConcepts,
  formatConceptStatus,
  formatMemoryLocation,
  summarizeMemorySnapshot,
} from './memoryPanelUtils.js'

test('formatConceptStatus maps memory statuses to Chinese labels', () => {
  assert.equal(formatConceptStatus('needs_review'), '需要复习')
  assert.equal(formatConceptStatus('learning'), '学习中')
  assert.equal(formatConceptStatus('familiar'), '较熟悉')
  assert.equal(formatConceptStatus('unknown'), '已记录')
})

test('summarizeMemorySnapshot creates stable panel counts', () => {
  const summary = summarizeMemorySnapshot({
    recent_events: [{ event_type: 'chat_question' }, { event_type: 'section_read' }],
    concepts: [{ concept: 'attention', status: 'needs_review' }],
    core: { current_position: 'doc:slide_2', session_summary: 'Recent learning actions: Asked a question' },
  })

  assert.deepEqual(summary, {
    currentPosition: 'Slide 3',
    sessionSummary: 'Recent learning actions: Asked a question',
    eventCount: 2,
    conceptCount: 1,
    reviewCount: 1,
  })
})

test('formatMemoryLocation hides raw ids and formats slide locations', () => {
  assert.equal(formatMemoryLocation({ section_id: 'slide_0', doc_id: '9f2d4a7c' }), 'Slide 1')
  assert.equal(formatMemoryLocation({ section_id: '', doc_id: '9f2d4a7c' }, '9f2d4a7c', 'lecture.pdf'), 'lecture.pdf')
  assert.equal(formatMemoryLocation({ section_id: '', doc_id: 'other_doc' }, '9f2d4a7c'), '其他文档')
})

test('filterReadableConcepts removes generated ids while keeping readable terms', () => {
  const concepts = filterReadableConcepts([
    { concept: 'a3f19c2b9e0d4c6a' },
    { concept: 'doc_01HZX9J2S2Q8V1A9B3C4D5E6F7' },
    { concept: 'self-attention' },
    { concept: 'Graph Neural Network' },
  ])

  assert.deepEqual(concepts.map(item => item.concept), ['self-attention', 'Graph Neural Network'])
})
