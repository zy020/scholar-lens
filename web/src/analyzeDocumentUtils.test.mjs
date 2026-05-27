import test from 'node:test'
import assert from 'node:assert/strict'
import { analyzeStatusText } from './analyzeDocumentUtils.js'

test('analyzeStatusText describes enhanced and unavailable results', () => {
  assert.equal(
    analyzeStatusText({ source: 'llm', status: 'enhanced' }),
    '增强分析完成',
  )
  assert.equal(
    analyzeStatusText({ source: 'unavailable', error: 'LLM is not configured.' }),
    'LLM is not configured.',
  )
  assert.equal(analyzeStatusText(null), '')
})
