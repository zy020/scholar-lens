import assert from 'node:assert/strict'
import test from 'node:test'
import { canInlinePreview, readerOpenLabel, readerPreviewUrl } from './readerPanelUtils.js'

test('reader utilities keep pdf inline preview behavior', () => {
  const doc = { name: 'paper.pdf', file_url: '/api/documents/abc/file' }

  assert.equal(canInlinePreview(doc), true)
  assert.equal(readerOpenLabel(doc), '新窗口打开 PDF')
})

test('reader utilities prefer explicit preview url when provided', () => {
  const doc = { name: 'lecture.pdf', file_url: '/api/documents/abc/file', preview_url: '/api/documents/abc/preview.pdf' }

  assert.equal(canInlinePreview(doc), true)
  assert.equal(readerPreviewUrl(doc), '/api/documents/abc/preview.pdf')
})
