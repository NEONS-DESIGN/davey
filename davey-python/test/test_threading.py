import sys
import sysconfig
import threading

import pytest
import davey

from test_session import create_session

FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
CONTROL_ITERATIONS = 2000
ENCRYPT_THREADS = 2
NON_SILENCE_FRAME = bytes(range(1, 101))


@pytest.mark.skipif(not FREE_THREADED_BUILD, reason="requires a free-threaded build")
def test_import_keeps_gil_disabled():
    assert not sys._is_gil_enabled()


def test_session_can_be_shared_between_threads():
    session = create_session(davey.SessionStatus.active)
    errors: list[BaseException] = []
    stop = threading.Event()

    def encrypt_loop():
        try:
            while not stop.is_set():
                session.encrypt_opus(NON_SILENCE_FRAME)
        except BaseException as e:
            errors.append(e)

    def control_loop():
        try:
            for _ in range(CONTROL_ITERATIONS):
                session.set_passthrough_mode(False, 10)
                session.get_encryption_stats()
                session.get_user_ids()
                repr(session)
                assert session.status == davey.SessionStatus.active
        except BaseException as e:
            errors.append(e)
        finally:
            stop.set()

    threads = [threading.Thread(target=encrypt_loop) for _ in range(ENCRYPT_THREADS)]
    threads.append(threading.Thread(target=control_loop))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    stats = session.get_encryption_stats()
    assert stats is not None
    assert stats.successes > 0
