import assert from 'node:assert/strict'
import test from 'node:test'
import { canInlinePreview, isPptxDocument, readerOpenLabel } from './readerPanelUtils.js'

test('reader utilities treat pptx as downloadable rather than inline pdf preview', () => {
  const doc = { name: 'lecture.pptx', file_url: '/api/documents/abc/file' }

  assert.equal(isPptxDocument(doc), true)
  assert.equal(canInlinePreview(doc), false)
  assert.equal(readerOpenLabel(doc), '下载 PPTX')
})

test('reader utilities keep pdf inline preview behavior', () => {
  const doc = { name: 'paper.pdf', file_url: '/api/documents/abc/file' }

  assert.equal(isPptxDocument(doc), false)
  assert.equal(canInlinePreview(doc), true)
  assert.equal(readerOpenLabel(doc), '新窗口打开 PDF')
})
