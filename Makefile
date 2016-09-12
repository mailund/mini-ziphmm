default:
	#pep8 mini_*.py
	python setup.py build_ext --inplace
clean:
	rm -rf build __pycache__ *.so mini_ziphmm_cython_funcs.c mini_ziphmm.egg-info *.pyc
