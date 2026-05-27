export function readerOpenLabel() {
  return '新窗口打开 PDF'
}

export function canInlinePreview() {
  return true
}

export function readerPreviewUrl(doc = {}) {
  return doc.preview_url || doc.file_url || ''
}
