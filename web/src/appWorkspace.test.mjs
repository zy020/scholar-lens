import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')

test('main workspace does not expose memory as a visible tab', () => {
  assert.doesNotMatch(source, /label: '记忆'/)
  assert.doesNotMatch(source, /<MemoryPanel/)
  assert.doesNotMatch(source, /from '\.\/MemoryPanel'/)
})
