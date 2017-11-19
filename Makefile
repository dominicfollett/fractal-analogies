segment-search:
	rm *.png | true
	PYOPENCL_CTX='0:1' python3 segment_search.py