from gpu_glm.models import backend_info


def test_backend_info_cpu(monkeypatch):
    """
    If CuPy is not available, backend_info should report CPU mode.
    """
    monkeypatch.setattr("gpu_glm.models.CUPY_AVAILABLE", False)
    out = backend_info()

    assert "NumPy (CPU)" in out
    assert "CuPy not installed" in out


def test_backend_info_gpu(monkeypatch):
    """
    If CuPy is available and GPU properties can be queried,
    backend_info should report GPU details.
    """

    class DummyProps(dict):
        pass

    class DummyRuntime:
        @staticmethod
        def getDevice():
            return 0

        @staticmethod
        def getDeviceProperties(device_id):
            return DummyProps(
                {
                    "name": b"Fake GPU",
                    "totalGlobalMem": 16 * 1024**3,
                    "multiProcessorCount": 42,
                }
            )

        @staticmethod
        def runtimeGetVersion():
            return 12040

        @staticmethod
        def driverGetVersion():
            return 12020

    class DummyCuPy:
        cuda = type("cuda", (), {"runtime": DummyRuntime})

    monkeypatch.setattr("gpu_glm.models.CUPY_AVAILABLE", True)
    monkeypatch.setattr("gpu_glm.models._cp", DummyCuPy)

    out = backend_info()

    assert "CuPy (GPU)" in out
    assert "Fake GPU" in out
    assert "16.00 GB" in out
    assert "Multiprocessors: 42" in out
    assert "CUDA Runtime Version: 12040" in out
    assert "CUDA Driver Version: 12020" in out


def test_backend_info_gpu_failure(monkeypatch):
    """
    If CuPy is available but GPU querying fails,
    backend_info should fall back gracefully.
    """

    class DummyRuntime:
        @staticmethod
        def getDevice():
            raise RuntimeError("GPU failure")

    class DummyCuPy:
        cuda = type("cuda", (), {"runtime": DummyRuntime})

    monkeypatch.setattr("gpu_glm.models.CUPY_AVAILABLE", True)
    monkeypatch.setattr("gpu_glm.models._cp", DummyCuPy)

    out = backend_info()

    assert "CuPy (GPU)" in out
    assert "could not be retrieved" in out
    assert "GPU failure" in out
