import numpy as np
import matplotlib.pyplot as plt

# Calculations

def calculate2DFT(img):
    fft_img = np.fft.ifftshift(img)
    fft_img = np.fft.fft2(fft_img)
    fft_img = np.fft.fftshift(fft_img)
    return fft_img

def calculateMagnitudSpectrum(img):
    #FFT
    img_fft = calculate2DFT(img)
    #Magnitud Spectrum
    img_ms = 20*np.log10(np.abs(img_fft))
    return img_ms

def calculate2DInverseFT(img_fft):
    # Calculate the inverse Fourier transform of 
    # the Fourier transform
    ift = np.fft.ifftshift(img_fft)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    ift = ift.real
    return ift

def cr8MaskForNoise(img,thresh_list=None):
    img_ms = calculateMagnitudSpectrum(img)
    
    if thresh_list == None:
        thresh_min = np.min(img_ms)
        thresh_max = np.max(img_ms)
    else:
        thresh_min = thresh_list[0]
        thresh_max = thresh_list[1]
    
    mask = img_ms
    mask[mask<thresh_min] = 1.0
    mask[mask>thresh_max] = 1.0
    mask[mask != 1.0] = 0.0
    
    return mask

def cr8MaskForSignal(img,thresh_list=None):
    img_ms = calculateMagnitudSpectrum(img)
    
    if thresh_list == None:
        thresh_min = np.min(img_ms)
        thresh_max = np.max(img_ms)
    else:
        thresh_min = thresh_list[0]
        thresh_max = thresh_list[1]
    
    mask = img_ms
    mask[mask<thresh_min] = 0.0
    mask[mask>thresh_max] = 0.0
    mask[mask != 0.0] = 1.0
    
    return mask

def getNoiseFFT(img,mask):
    
    img_fft = calculate2DFT(img)
    
    noise = calculate2DInverseFT(img_fft*mask)

    noise = np.uint8(noise)
    
    return noise

def denoisingFFT(img,ths_list,mode='1'):
    
    if mode == '0':
    
        mask = cr8MaskForNoise(img,ths_list)
    
        img_fft = calculate2DFT(img)
        
        noise = calculate2DInverseFT(img_fft*mask)
        
        dummy0 = img - noise

        dummy1 = dummy0

        dummy1[dummy1 < 0] = 0

        dummy2 = np.uint8(dummy1)
        
        return dummy2
    
    elif mode == '1':
        mask = cr8MaskForSignal(img,ths_list)
        
        img_fft = calculate2DFT(img)
        
        signal = calculate2DInverseFT(img_fft*mask)
        
        signal[signal < 0] = 0
        
        signal = np.uint8(signal)
        
        return signal


# Plots

def cr8FFTimgsPlots(img,name='name'):
    #Magnitud Spectrum
    img_ms = calculateMagnitudSpectrum(img)
    #3D plots parameters

    min_ms = np.min(img_ms)
    max_ms = np.max(img_ms)

    alto, ancho = np.shape(img_ms)

    X = np.arange(ancho)
    Y = np.arange(alto)
    X,Y = np.meshgrid(X,Y)
    Z = img_ms

    # 0º - Figure set-up
    fig = plt.figure(figsize=(22,12))

    # 1º subplot
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(img,vmin=0,vmax=255,cmap="gray")
    ax.set_title(f"Original - {name}")

    # 2º subplot
    ax = fig.add_subplot(2, 3, 2)
    gray_ms = ax.imshow(img_ms,vmin=min_ms,vmax=max_ms,cmap='gray')
    ax.set_title("Magnitud espectral")

    # 3º subplot
    ax = fig.add_subplot(2, 3, 3)
    color_ms = ax.imshow(img_ms,vmin=min_ms,vmax=max_ms,cmap='gnuplot2')
    ax.set_title("Magnitud espectral")

    # 4º subplot
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    surf1 = ax.plot_surface(X, Y, Z,vmin=min_ms,vmax=max_ms,cmap="gnuplot2")
    ax.set_title("Magnitud espectral - Vista isométrica")
    ax.set_xlabel('Eje x')
    ax.set_ylabel('Eje y')
    ax.set_zlabel('Magnitud')

    # 5º subplot
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    surf2 = ax.plot_surface(X, Y, Z,vmin=min_ms,vmax=max_ms,cmap="gnuplot2")
    ax.view_init(elev=0, azim=-90, roll=0)
    ax.set_title("Magnitud espectral - Vista XZ")
    ax.set_yticks([])
    ax.set_xlabel('Eje x')
    ax.set_zlabel('Magnitud')

    # 5º subplot
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    surf3 = ax.plot_surface(X, Y, Z,vmin=min_ms,vmax=max_ms,cmap="gnuplot2")
    ax.view_init(elev=0, azim=0, roll=0)
    ax.set_title("Magnitud espectral - Vista YZ")
    ax.set_xticks([])
    ax.set_ylabel('Eje y')
    ax.set_zlabel('Magnitud')

    #Colorbars
    fig.colorbar(gray_ms,shrink=0.75,aspect=10,label='dB')
    fig.colorbar(color_ms,shrink=0.75,aspect=10,label='dB')
    fig.colorbar(surf1,shrink=0.75,aspect=10,label='dB')
    fig.colorbar(surf2,shrink=0.75,aspect=10,label='dB')
    fig.colorbar(surf3,shrink=0.75,aspect=10,label='dB')

    plt.show()

    return fig

def cr8FFTNoiseEstimPlots(img,thresh_list,name='name'):
    mask = cr8MaskForNoise(img,thresh_list)
    noise = getNoiseFFT(img,mask)
    
    min_mk = np.min(mask)
    max_mk = np.max(mask)

    alto, ancho = np.shape(mask)

    X = np.arange(ancho)
    Y = np.arange(alto)
    X,Y = np.meshgrid(X,Y)
    Z = mask

    # 0º - Figure set-up
    fig = plt.figure(figsize=(22,6))

    # 1º subplot
    ax = fig.add_subplot(1, 3, 1)
    color_mk = ax.imshow(mask,vmin=min_mk,vmax=max_mk,cmap='gnuplot2')
    ax.set_title(f'Máscara - {name}')

    # 2º subplot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    surf1 = ax.plot_surface(X, Y, Z,vmin=min_mk,vmax=max_mk,cmap='gnuplot2')
    ax.set_title("Máscara - Vista isométrica")
    ax.set_xlabel('Eje x')
    ax.set_ylabel('Eje y')

    # 3º subplot
    ax = fig.add_subplot(1, 3, 3)
    gray_ms = ax.imshow(noise,vmin=0,vmax=255,cmap='gray')
    ax.set_title(f"Ruido estimado - {name}")

    #Colorbars
    fig.colorbar(color_mk,shrink=0.75,aspect=10)
    fig.colorbar(surf1,shrink=0.75,aspect=10)

    plt.show()

    return fig
    
def cr8FFTDenoisingPlots(img,thresh_list,name='name'):
    
    filtered_img = denoisingFFT(img,thresh_list)

    # 0º - Figure set-up
    fig = plt.figure(figsize=(14,7))

    # 1º subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img,vmin=0,vmax=255,cmap='gray')
    ax.set_title(f'Original - {name}')

    # 2º subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(filtered_img,vmin=0,vmax=255,cmap='gray')
    ax.set_title(f"Denoising - {name}")

    plt.show()

    return filtered_img