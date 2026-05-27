import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

const source = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')

test('workspace tab names the former study brief area as document analysis', () => {
  assert.match(source, /\{\s*id:\s*'study',\s*label:\s*'文档分析'\s*\}/)
  assert.doesNotMatch(source, /\{\s*id:\s*'study',\s*label:\s*'学习简报'\s*\}/)
})
