import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    void mask(int M, int N, int chan, double dtth, double eps, int* tam, double* ta, double* band, double* tths, double* omask);
    """)

ffibuilder.set_source("_mask", "", sources=["mask.cpp"],)

ffibuilder.compile(verbose=True)
