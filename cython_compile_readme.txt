0. download and install vc++ express for python compiler
1. open visual c++ 2008 64-bit command prompt (for 64-bit python 2.7)
2. go to the folder where setup.py and .pyx is located at
3. type in: set distutils_use_sdk=1
4. type in: set mssdk=1
5. type in: python.exe setup.py build_ext --inplace --compiler=msvc

