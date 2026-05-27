import { canInlinePreview, readerOpenLabel } from './readerPanelUtils'

export default function ReaderPanel({ doc }) {
  const fileUrl = doc.file_url || null

  if (!doc.name) {
    return <div className="reader-panel"><p className="empty">请先上传文档</p></div>
  }

  const inlinePreview = canInlinePreview(doc)

  return (
    <div className="reader-panel">
      <div className="reader-header">
        <span>{doc.name}</span>
        <div className="reader-actions">
          {fileUrl && (
            <a href={fileUrl} target="_blank" rel="noreferrer" className="pdf-link">{readerOpenLabel(doc)}</a>
          )}
        </div>
      </div>
      {fileUrl && inlinePreview && <iframe src={fileUrl} className="pdf-viewer" title="文档预览" />}
      {fileUrl && !inlinePreview && <div className="pdf-placeholder">PPTX 可在右侧章节与问答中学习，原文件请通过上方按钮下载。</div>}
      {!fileUrl && <div className="pdf-placeholder">原文件不可用</div>}
    </div>
  )
}
