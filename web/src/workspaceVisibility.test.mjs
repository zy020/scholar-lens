import assert from 'node:assert/strict'
import test from 'node:test'
import { canShowWorkspace } from './workspaceVisibility.js'

test('config workspace remains available before uploading documents', () => {
  assert.equal(canShowWorkspace({ hasDoc: false, tab: 'config' }), true)
  assert.equal(canShowWorkspace({ hasDoc: false, tab: 'chat' }), false)
  assert.equal(canShowWorkspace({ hasDoc: true, tab: 'chat' }), true)
})
