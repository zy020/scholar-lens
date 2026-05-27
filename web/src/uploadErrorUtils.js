export function formatUploadError(fileName, uploadKind, message) {
  const detail = message || '未知错误'
  if (uploadKind === 'paper' && /pdf files only|pdf/i.test(detail)) {
    return `${fileName} 上传失败: 论文上传仅支持 PDF 文件`
  }
  if (uploadKind === 'courseware' && /pdf/i.test(detail)) {
    return `${fileName} 上传失败: 课件上传仅支持 PDF 文件`
  }
  return `${fileName} 上传失败: ${detail}`
}
