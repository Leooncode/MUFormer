import matplotlib.pyplot as plt

def plot_abundance(abu, iter_rec, loss_rec):
    f, ([ax1, ax2, ax3, ax4, ax5, ax6], [ax7, ax8, ax9, ax10, ax11, ax12],
        [ax13, ax14, ax15, ax16, ax17, ax18]) = plt.subplots(3, 6, sharey=False, figsize=(10,10))
    
    ax1.imshow(abu[:, :, 0, 0], cmap='jet')
    ax7.imshow(abu[:, :, 1, 0], cmap='jet')
    ax13.imshow(abu[:, :, 2, 0], cmap='jet')
    
    ax2.imshow(abu[:, :, 0, 1], cmap='jet')
    ax8.imshow(abu[:, :, 1, 1], cmap='jet')
    ax14.imshow(abu[:, :, 2, 1], cmap='jet')
    
    ax3.imshow(abu[:, :, 0, 2], cmap='jet')
    ax9.imshow(abu[:, :, 1, 2], cmap='jet')
    ax15.imshow(abu[:, :, 2, 2], cmap='jet')
    
    ax4.imshow(abu[:, :, 0, 3], cmap='jet')
    ax10.imshow(abu[:, :, 1, 3], cmap='jet')
    ax16.imshow(abu[:, :, 2, 3], cmap='jet')
    
    ax5.imshow(abu[:, :, 0, 4], cmap='jet')
    ax11.imshow(abu[:, :, 1, 4], cmap='jet')
    ax17.imshow(abu[:, :, 2, 4], cmap='jet')
    
    ax6.imshow(abu[:, :, 0, 5], cmap='jet')
    ax12.imshow(abu[:, :, 1, 5], cmap='jet')
    ax18.imshow(abu[:, :, 2, 5], cmap='jet')

    plt.show()
    
    plt.subplot(111).plot(iter_rec, loss_rec)
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")
    plt.show() 