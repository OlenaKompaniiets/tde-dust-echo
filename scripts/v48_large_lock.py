"""Cross-process cache lock, released by OS after interruption."""
import contextlib,os,time
@contextlib.contextmanager
def cache_lock(path):
    f=path.open('a+b')
    if path.stat().st_size==0:f.write(b'0');f.flush()
    try:
        while True:
            f.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:time.sleep(.25)
        yield
    finally:f.close()
