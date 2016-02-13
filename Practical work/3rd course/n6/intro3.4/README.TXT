Please follow the next instructions for building and testing examples!

1) Run
`python3.4 setup.py build_ext --inplace`
which builds the directory 'build' with auxiliary files and 'fast.so' file
you can use in 'import'-section of python-programs.
2) Run
'python3.4 test.py'
which runs several algorithms from fast.pyx and slow.py
