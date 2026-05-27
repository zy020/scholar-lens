import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(new URL('./ChatPanel.jsx', import.meta.url), 'utf8')

test('ChatPanel exposes deep mode and sends it with chat requests', () => {
  assert.match(source, /deepMode/)
  assert.match(source, /深度模式/)
  assert.match(source, /deep_mode:\s*deepMode/)
})

test('ChatPanel does not render education-unfriendly evidence cards', () => {
  assert.doesNotMatch(source, /evidence-card/)
  assert.doesNotMatch(source, /原文证据/)
  assert.doesNotMatch(source, /getCitedEvidenceView/)
})
