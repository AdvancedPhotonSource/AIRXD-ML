import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void mask(int M, int N, int chan, double dtth, double eps, int* tam, double* ta, double* band, double* tths, double* omask);
    """)

ffibuilder.set_source("airxd._mask", "", sources=["airxd/mask.cpp"], include_dirs=["airxd/"])

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
