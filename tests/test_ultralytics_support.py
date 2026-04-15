import sys
import types
import unittest
import unittest.mock

from yolo_frigate.ultralytics_support import ensure_tensorrt_namespace


class TestUltralyticsSupport(unittest.TestCase):
    def test_ensure_tensorrt_namespace_aliases_bindings_package(self):
        bindings = types.ModuleType("tensorrt_bindings")
        bindings.__version__ = "10.16.1.11"
        bindings.Builder = object()

        plugin = types.ModuleType("tensorrt_bindings.plugin")
        plugin.__path__ = []
        plugin_autotune = types.ModuleType("tensorrt_bindings.plugin._autotune")
        plugin_autotune.marker = "autotune"

        with unittest.mock.patch.dict(
            sys.modules,
            {
                "tensorrt_bindings": bindings,
                "tensorrt_bindings.plugin": plugin,
                "tensorrt_bindings.plugin._autotune": plugin_autotune,
            },
        ):
            sys.modules.pop("tensorrt", None)
            sys.modules.pop("tensorrt.plugin", None)
            sys.modules.pop("tensorrt.plugin._autotune", None)

            ensure_tensorrt_namespace()

            self.assertEqual(sys.modules["tensorrt"].__version__, "10.16.1.11")
            self.assertIs(sys.modules["tensorrt.plugin"], plugin)
            self.assertIs(sys.modules["tensorrt.plugin._autotune"], plugin_autotune)

    def test_ensure_tensorrt_namespace_noops_without_bindings(self):
        with unittest.mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("tensorrt", None)
            sys.modules.pop("tensorrt_bindings", None)

            ensure_tensorrt_namespace()

            self.assertNotIn("tensorrt", sys.modules)
