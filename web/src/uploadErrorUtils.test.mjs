import assert from 'node:assert/strict'
import test from 'node:test'
import { formatUploadError } from './uploadErrorUtils.js'

test('paper upload extension error explains PDF-only entrance', () => {
  const message = formatUploadError('slides.pptx', 'paper', 'Paper uploads accept PDF files only')

  assert.match(message, /slides\.pptx/)
  assert.match(message, /论文上传仅支持 PDF/)
})

test('courseware upload extension error explains PDF and PPTX support', () => {
  const message = formatUploadError('notes.docx', 'courseware', 'Courseware uploads accept PDF or PPTX files only')

  assert.match(message, /notes\.docx/)
  assert.match(message, /课件上传支持 PDF\/PPTX/)
})

test('generic upload error keeps original message', () => {
  const message = formatUploadError('paper.pdf', 'paper', 'HTTP 500')

  assert.equal(message, 'paper.pdf 上传失败: HTTP 500')
})
