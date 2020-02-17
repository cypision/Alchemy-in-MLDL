# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x): #호출한 W (2,3)
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h #호출한  x = W (2,3) 의 값을 근사하게 바꿈 그때의 costFunction 값
        print('인수는2개인데x만 간다',x)
        fxh1 = f(x) # f(x+h)        #근사하게 바꾼 결과값
        
        x[idx] = tmp_val - h  #호출한 W (2,3) 의 값을 근사하게 바꿈 그때의 costFunction 값
        fxh2 = f(x) # f(x-h)  #근사하게 바꾼 결과값
        grad[idx] = (fxh1 - fxh2) / (2*h) # 차분한다.
        
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)   # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad
