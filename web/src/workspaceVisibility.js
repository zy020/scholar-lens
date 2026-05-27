export function canShowWorkspace({ hasDoc = false, tab = 'chat' } = {}) {
  return hasDoc || tab === 'config'
}
