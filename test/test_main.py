#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
import unittest

import mox


class TestMain(unittest.TestCase):

  def setUp(self):
    self.mox = mox.Mox()
    self.rime = rime.Rime()
    self.options = self.rime.GetDefaultOptions()
    self.mock_ctx = self.mox.CreateMock(rime.RimeContext)

  def tearDown(self):
    pass

  def testLoadRoot_Success(self):
    mock_root = self.mox.CreateMock(rime.RimeRoot)
    self.mox.StubOutWithMock(rime, 'RimeRoot', use_mock_anything=True)
    rime.RimeRoot.CanLoadFrom("/hoge/piyo").AndReturn(False)
    rime.RimeRoot.CanLoadFrom("/hoge").AndReturn(True)
    rime.RimeRoot(None, "/hoge", None, self.options, self.mock_ctx).AndReturn(mock_root)
    self.mox.ReplayAll()
    self.assertEquals(mock_root, self.rime.LoadRoot("/hoge/piyo", self.options, self.mock_ctx))
    self.mox.VerifyAll()

  def testLoadRoot_Failure(self):
    mock_root = self.mox.CreateMock(rime.RimeRoot)
    self.mox.StubOutWithMock(rime, 'RimeRoot', use_mock_anything=True)
    rime.RimeRoot.CanLoadFrom("/hoge/piyo").AndReturn(False)
    rime.RimeRoot.CanLoadFrom("/hoge").AndReturn(False)
    rime.RimeRoot.CanLoadFrom("/").AndReturn(False)
    self.mox.ReplayAll()
    self.assertEquals(None, self.rime.LoadRoot("/hoge/piyo", self.options, self.mock_ctx))
    self.mox.VerifyAll()


if __name__ == '__main__':
  base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
  sys.path.insert(0, base_dir)
  import rime
  unittest.main()
