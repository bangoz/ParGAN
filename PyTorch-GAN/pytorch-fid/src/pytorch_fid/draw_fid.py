import os, sys

# path1 = '/data1/xiechuhan/PyTorch-GAN/implementations/_par_gan/images_for_fid/try3/0.pt'
# path2 = '/data1/xiechuhan/datasets/MNIST/processed/training.pt'

# output = os.popen('/data1/xiechuhan/anaconda3/bin/python3.8 /data1/xiechuhan/PyTorch-GAN/pytorch-fid/src/pytorch_fid/fid_score.py {} {}'.format(path1, path2)).read()

# print(output)

def model_fids(path, epochs=200):
    fids = []
    for epoch in range(0,epochs,20):
        path1 = path.format(epoch)
        path2 = '/data1/xiechuhan/datasets/MNIST/processed/training.pt'
        output = os.popen('/data1/xiechuhan/anaconda3/bin/python3.8 /data1/xiechuhan/PyTorch-GAN/pytorch-fid/src/pytorch_fid/fid_score.py {} {}'.format(path1, path2)).read()
        print(output)

        fids.append(float(output.split(' ')[-1]))
    
    return fids
    
paths = ['/data1/xiechuhan/PyTorch-GAN/implementations/wgan_gp/images_for_fid/try1/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/_par_wgan_gp/images_for_fid/{}.pt'][1:]
# paths = ['/data1/xiechuhan/PyTorch-GAN/implementations/dcgan/images_for_fid/try1/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/_par_dcgan/images_for_fid/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/lsgan/images_for_fid/try1/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/_par_lsgan/images_for_fid/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/wgan/images_for_fid/try1/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/_par_wgan/images_for_fid/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/wgan_gp/images_for_fid/try1/{}.pt', '/data1/xiechuhan/PyTorch-GAN/implementations/_par_wgan_gp/images_for_fid/try1/{}.pt']

fids_dict = {}
for i, path in enumerate(paths):
    fids_dict[i] = model_fids(path)
    print(i, 'done.')


### 画图的代码
# plt.figure(figsize=(8,12), dpi=80)
# plt.figure(1)
# plt.subplots_adjust(wspace=0.25, hspace=0.25)

# ax1 = plt.subplot(321)
# ax1.plot(range(0,200,20),a1[0][:10], color="orange",linestyle = "--")
# ax1.plot(range(0,200,20),a1[1][:10], color="xkcd:turquoise",linestyle = "-")
# ax1.legend(['GAN','ParGAN'])
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('FID')
# ax1.set_title('GAN & ParGAN')

# ax2 = plt.subplot(322)
# ax2.plot(range(0,200,20),fids_dict[0], color="orange",linestyle = "--")
# ax2.plot(range(0,200,20),fids_dict[1], color="xkcd:turquoise",linestyle = "-")
# ax2.legend(['DCGAN','ParDCGAN'])
# ax2.set_xlabel('epochs')
# ax2.set_ylabel('FID')
# ax2.set_title('DCGAN & ParDCGAN')

# ax3 = plt.subplot(323)
# ax3.plot(range(0,200,20),fids_dict[2], color="orange",linestyle = "--")
# ax3.plot(range(0,200,20),fids_dict[3], color="xkcd:turquoise",linestyle = "-")
# ax3.legend(['LSGAN','ParLSGAN'])
# ax3.set_xlabel('epochs')
# ax3.set_ylabel('FID')
# ax3.set_title('LSGAN & ParLSGAN')

# ax4 = plt.subplot(324)
# ax4.plot(range(0,200,20),fids_dict[4], color="orange",linestyle = "--")
# ax4.plot(range(0,200,20),fids_dict[5], color="xkcd:turquoise",linestyle = "-")
# ax4.legend(['WGAN','ParWGAN'])
# ax4.set_xlabel('epochs')
# ax4.set_ylabel('FID')
# ax4.set_title('WGAN & ParWGAN')

# ax5 = plt.subplot(325)
# ax5.plot(range(0,200,20),fids_dict[6], color="orange",linestyle = "--")
# ax5.plot(range(0,200,20),fids_dict[7], color="xkcd:turquoise",linestyle = "-")
# ax5.legend(['WGAN-GP','ParWGAN-GP'])
# ax5.set_xlabel('epochs')
# ax5.set_ylabel('FID')
# ax5.set_title('WGAN-GP & ParWGAN-GP')