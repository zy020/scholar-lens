export function isPptxDocument(doc = {}) {
  const name = String(doc.name || '').toLowerCase()
  const fileUrl = String(doc.file_url || '').toLowerCase()
  return name.endsWith('.pptx') || fileUrl.endsWith('.pptx')
}

export function readerOpenLabel(doc = {}) {
  return isPptxDocument(doc) ? '下载 PPTX' : '新窗口打开 PDF'
}

export function canInlinePreview(doc = {}) {
  return !isPptxDocument(doc)
}
