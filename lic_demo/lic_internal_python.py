import numpy as np

def advance(vx, vy, x, y, fx, fy, w, h):
    if vx>=0:
        tx = (1-fx)/vx
    else:
        tx = -fx/vx
    if vy>=0:
        ty = (1-fy)/vy
    else:
        ty = -fy/vy
    if tx<ty:
        if vx>=0:
            x+=1
            fx=0
        else:
            x-=1
            fx=1
        fy+=tx*vy
    else:
        if vy>=0:
            y+=1
            fy=0
        else:
            y-=1
            fy=1
        fx+=ty*vx
    if x>=w:
        x=w-1 # FIXME: other boundary conditions?
    if x<0:
        x=0 # FIXME: other boundary conditions?
    if y<0:
        y=0 # FIXME: other boundary conditions?
    if y>=h:
        y=h-1 # FIXME: other boundary conditions?
    return x, y, fx, fy, tx, ty


#np.ndarray[float, ndim=2]
def line_integral_convolution(vectors, texture, kernel):
    h = vectors.shape[0]
    w = vectors.shape[1]
    kernellen = kernel.shape[0]
    result = np.zeros((h,w),dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x = j
            y = i
            fx = 0.5
            fy = 0.5
            
            k = kernellen//2
            #print i, j, k, x, y
            result[i,j] += kernel[k]*texture[x,y]
            while k<kernellen-1:
                x,y,fx,fy,tx,ty = advance(vectors[y,x,0],vectors[y,x,1],x,y,fx,fy,w,h)
                k+=1
                result[i,j] += kernel[k]*texture[x,y]

            x = j
            y = i
            fx = 0.5
            fy = 0.5
            
            while k>0:
                x,y,fx,fy,tx,ty = advance(-vectors[y,x,0],-vectors[y,x,1],x,y,fx,fy,w,h)
                k-=1
                result[i,j] += kernel[k]*texture[x,y]
                    
    return result

