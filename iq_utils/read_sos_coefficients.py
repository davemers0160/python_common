import numpy as np

#------------------------------------------------------------------------------
# Define a function to convert the input string to a complex number
def complex_converter(s):
    v = complex(s.decode('utf-8').replace('i', 'j'))
    # v = complex(s.replace('i', 'j'))
    return v

#------------------------------------------------------------------------------
def read_complex_sos_coefficients(filename: str):
    # Read the CSV using genfromtxt with the converter
    data = np.genfromtxt(filename, delimiter=',', dtype=np.complex128, converters={idx: complex_converter for idx in range(6)} )
    # data = np.genfromtxt(filename, delimiter=',', dtype=np.complex128, converters={0: complex_converter} )

    # print(data)
    
    return data

#------------------------------------------------------------------------------  
def read_sos_coefficients(filename: str):

    data = np.loadtxt('data.csv', delimiter=',').as_type(np.float64)
    
    # print(data)
    
    return data
