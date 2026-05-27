import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./Sidebar.jsx', import.meta.url), 'utf8')

test('Sidebar exposes one PDF upload entry with explicit document kind selection', () => {
  assert.match(source, /论文/)
  assert.match(source, /课件/)
  assert.match(source, /accept="\.pdf"/)
  assert.match(source, /disabled=\{uploading\}/)
  assert.doesNotMatch(source, /\.pptx|PPTX/)
  assert.doesNotMatch(source, /上传论文 PDF[\s\S]*上传课件 PDF/)
})
