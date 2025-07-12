import matplotlib.pyplot as plt


def plot_abundance(abu, iter_rec, loss_rec):
    f, ([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15], \
        [ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29, ax30],\
        [ax31, ax32, ax33, ax34, ax35, ax36, ax37, ax38, ax39, ax40, ax41, ax42, ax43, ax44, ax45], \
        [ax46, ax47, ax48, ax49, ax50, ax51, ax52, ax53, ax54, ax55, ax56, ax57, ax58, ax59, ax60]) = plt.subplots(4, 15, sharey=False, figsize=(30, 30))
    
    ax1.imshow(abu[:, :, 0, 0], cmap='jet')
    ax16.imshow(abu[:, :, 1, 0], cmap='jet')
    ax31.imshow(abu[:, :, 2, 0], cmap='jet')
    ax46.imshow(abu[:, :, 3, 0], cmap='jet')
    
    ax2.imshow(abu[:, :, 0, 1], cmap='jet')
    ax17.imshow(abu[:, :, 1, 1], cmap='jet')
    ax32.imshow(abu[:, :, 2, 1], cmap='jet')
    ax47.imshow(abu[:, :, 3, 1], cmap='jet')
    
    ax3.imshow(abu[:, :, 0, 2], cmap='jet')
    ax18.imshow(abu[:, :, 1, 2], cmap='jet')
    ax33.imshow(abu[:, :, 2, 2], cmap='jet')
    ax48.imshow(abu[:, :, 3, 2], cmap='jet')
    
    ax4.imshow(abu[:, :, 0, 3], cmap='jet')
    ax19.imshow(abu[:, :, 1, 3], cmap='jet')
    ax34.imshow(abu[:, :, 2, 3], cmap='jet')
    ax49.imshow(abu[:, :, 3, 3], cmap='jet')
    
    ax5.imshow(abu[:, :, 0, 4], cmap='jet')
    ax20.imshow(abu[:, :, 1, 4], cmap='jet')
    ax35.imshow(abu[:, :, 2, 4], cmap='jet')
    ax50.imshow(abu[:, :, 3, 4], cmap='jet')
    
    ax6.imshow(abu[:, :, 0, 5], cmap='jet')
    ax21.imshow(abu[:, :, 1, 5], cmap='jet')
    ax36.imshow(abu[:, :, 2, 5], cmap='jet')
    ax51.imshow(abu[:, :, 3, 5], cmap='jet')
    
    ax7.imshow(abu[:, :, 0, 6], cmap='jet')
    ax22.imshow(abu[:, :, 1, 6], cmap='jet')
    ax37.imshow(abu[:, :, 2, 6], cmap='jet')
    ax52.imshow(abu[:, :, 3, 6], cmap='jet')
    
    ax8.imshow(abu[:, :, 0, 7], cmap='jet')
    ax23.imshow(abu[:, :, 1, 7], cmap='jet')
    ax38.imshow(abu[:, :, 2, 7], cmap='jet')
    ax53.imshow(abu[:, :, 3, 7], cmap='jet')
    
    ax9.imshow(abu[:, :, 0, 8], cmap='jet')
    ax24.imshow(abu[:, :, 1, 8], cmap='jet')
    ax39.imshow(abu[:, :, 2, 8], cmap='jet')
    ax54.imshow(abu[:, :, 3, 8], cmap='jet')
    
    ax10.imshow(abu[:, :, 0, 9], cmap='jet')
    ax25.imshow(abu[:, :, 1, 9], cmap='jet')
    ax40.imshow(abu[:, :, 2, 9], cmap='jet')
    ax55.imshow(abu[:, :, 3, 9], cmap='jet')
    
    ax11.imshow(abu[:, :, 0, 10], cmap='jet')
    ax26.imshow(abu[:, :, 1, 10], cmap='jet')
    ax41.imshow(abu[:, :, 2, 10], cmap='jet')
    ax56.imshow(abu[:, :, 3, 10], cmap='jet')
    
    ax12.imshow(abu[:, :, 0, 11], cmap='jet')
    ax27.imshow(abu[:, :, 1, 11], cmap='jet')
    ax42.imshow(abu[:, :, 2, 11], cmap='jet')
    ax57.imshow(abu[:, :, 3, 11], cmap='jet')
    
    ax13.imshow(abu[:, :, 0, 12], cmap='jet')
    ax28.imshow(abu[:, :, 1, 12], cmap='jet')
    ax43.imshow(abu[:, :, 2, 12], cmap='jet')
    ax58.imshow(abu[:, :, 3, 12], cmap='jet')
    
    ax14.imshow(abu[:, :, 0, 13], cmap='jet')
    ax29.imshow(abu[:, :, 1, 13], cmap='jet')
    ax44.imshow(abu[:, :, 2, 13], cmap='jet')
    ax59.imshow(abu[:, :, 3, 13], cmap='jet')
    
    ax15.imshow(abu[:, :, 0, 14], cmap='jet')
    ax30.imshow(abu[:, :, 1, 14], cmap='jet')
    ax45.imshow(abu[:, :, 2, 14], cmap='jet')
    ax60.imshow(abu[:, :, 3, 14], cmap='jet')
    plt.show()
    
    plt.subplot(111).plot(iter_rec, loss_rec)
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")
    plt.show() 