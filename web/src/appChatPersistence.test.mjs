import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')

test('chat panel is not remounted when switching documents', () => {
  assert.match(source, /<ChatPanel\s+doc=\{activeDoc\}/)
  assert.doesNotMatch(source, /<ChatPanel\s+key=\{activeDoc\.doc_id\}/)
})
