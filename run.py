from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from torch.utils.data import DataLoader
from model import *
from DeepLearningUtils.data import *


IMG_SHAPE= [256,192]

def train(num_epochs, latent_dims, path, device, data_loader_train, data_loader_val, diff_fac=10, loss_type="L1",
          smooth=True, fixed_atlas=None):
    # fix the atlas?
    if fixed_atlas is None:
        init_template = torch.mean(torch.cat(data_loader_train.dataset.images).unsqueeze(1).to(device), dim=0)
        fixed_flag=False
    else:
        img = np.array(Image.open(fixed_atlas).resize((IMG_SHAPE[1],IMG_SHAPE[0]))) / 255.
        init_template=torch.tensor(img).unsqueeze(0).float().to(device)
        fixed_flag=True

    model = ADAE(latent_dim_a=latent_dims[0],latent_dim_s=latent_dims[1], img_dim=IMG_SHAPE, template=init_template,
                 smooth=smooth, fixed=fixed_flag)
    model.to(device)
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_ls_train = []
    loss_ls_test = []

    best_loss = 10000000
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        val_loss=0
        for batch_idx, data in enumerate(data_loader_train):
            img_batch = data[0].to(device)

            optimizer.zero_grad()
            displ_field_run, recon_batch, mu_run, logvar_run = model(img_batch)

            loss = loss_function(recon_img=recon_batch, input_img=img_batch, disp_field=displ_field_run, mu=mu_run,
                                 logvar=logvar_run, loss_type=loss_type,diff_fac=diff_fac)
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()

        model.eval()
        for batch_idx, data_val in enumerate(data_loader_val):
            img_batch = data_val[0].to(device)

            optimizer.zero_grad()
            displ_field_val, recon_batch_val, mu_run_val, logvar_run_val = model(img_batch)

            loss = loss_function(recon_img=recon_batch_val, input_img=img_batch, disp_field=displ_field_val, mu=mu_run_val,
                                 logvar=logvar_run_val, loss_type=loss_type)
            val_loss+= loss.data.item()

        loss_ls_test.append(val_loss/len(data_loader_val.dataset))
        loss_ls_train.append(train_loss/len(data_loader_val.dataset))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_loader_train.dataset)))

        if (epoch+1) %10==0:
            plt.plot(loss_ls_test, 'r', label='Loss Test')
            plt.plot(loss_ls_train, 'b', label='Loss Train')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(path + '/learning_progress.png')
            plt.close()
            if loss_ls_test[epoch]<=best_loss:
                best_loss=loss_ls_test[epoch]
                torch.save(model, path + '/DAE_model.pt')

            txt = open(path + '/about_model.txt', 'w')
            txt.write('Model: ' + str(latent_dims) + ' Dims \n'
                                                    'Average Loss: ' + str(
                train_loss / len(data_loader_train.dataset)) + ' nach ' + str(epoch) + ' epochen \n' +
                      'latent_dim: ' + str(latent_dims) + '\n' +
                      'learning_rate: ' + str(lr) + '\n' +
                      'batch size: ' + str(data[0].shape[0]) + '\n' +
                      'diffusion factor: ' + str(diff_fac) + '\n'
                      )
            txt.close()
    del model

def test(latent_dims, path, device, data_loader_test):

    model_test = torch.load(path + '/DAE_model.pt',map_location=torch.device(device))

    model_test.eval()
    test_loss_sum = 0

    print('\n' + '###### Test ######' + '\n')

    for idx, data in enumerate(data_loader_test):
        img_batch = data[0].to(device)

        displ_field_run, recon_batch, mu_run, logvar_run, = model_test(img_batch)

        loss = loss_function(recon_img=recon_batch, input_img=img_batch, disp_field=displ_field_run, mu=mu_run,
                             logvar=logvar_run)

        print('Loss Image ' + str(idx) + ': ' + str(loss.data.item()))
        plt.figure(figsize=(4 * 3, 4))
        G = grsp.GridSpec(1, 3)
        img_rec = np.array((recon_batch).float().cpu().squeeze().detach())
        img_real = np.array((img_batch.cpu().squeeze().detach()))
        img_delta_a = np.array((model_test.diff_a.cpu().squeeze().detach()))
        ax0 = plt.subplot(G[0, 0])
        ax1 = plt.subplot(G[0, 1])
        ax2 = plt.subplot(G[0, 2])
        ax0.imshow(img_real, cmap="gray")
        ax1.imshow(img_rec, cmap="gray")
        ax2.imshow(img_delta_a, cmap="gray")
        plt.savefig(path + '/rec_img' + str(idx) + '.png')
        test_loss_sum += loss.data.item()

    print('\n' + 'Loss mean: ' + str(test_loss_sum/idx))

    txt = open(path + '/about_model.txt', 'a')
    txt.write('\nAverage Loss Test: ' + str(test_loss_sum/idx))

    # Generate new Images:
    generate_imges(num_imgs=5, model_test=model_test, latent_dim=latent_dims[0]+latent_dims[1], path=path,
                   train_data=torch.cat(train_data.images).unsqueeze(1))


    interpolate_images(test_data.images[274].unsqueeze(0).to(device), test_data.images[275].unsqueeze(0).to(device),
                       model_test,path=path, type="app")
    interpolate_images(test_data.images[274].unsqueeze(0).to(device), test_data.images[275].unsqueeze(0).to(device),
                       model_test, path=path, type="shape")
    txt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help='Nr of epochs (a large number about 1000)')
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--device_nr", type=int, default=1, help="GPU number")
    parser.add_argument("--n_training", type=int, default=500, help="Number of training images, 0 if all samples are to be taken")
    parser.add_argument("--n_folds", type=int, default=1, help="Number of folds")
    parser.add_argument("--test",type=bool, default=False, help="Test?")
    parser.add_argument("--train", type=bool, default=False, help="Train?")
    parser.add_argument("--loss", type=str, default="SSIM", help="What loss?")
    parser.add_argument("--smooth", type=bool, default=True, help="Use GF smoothin?")
    parser.add_argument("--fixed_atlas", type=str, help="path to fixed atlas")
    parser.add_argument("--modality", type=str, default="all", help="all, t1, t2")
    parser.add_argument("--hospital", type=str, default="all", help="all, [guys, hh, iop]<-- all combinations")
    parser.add_argument("--latent_dim_a", type=int, default=64, help="appearance latent dim")
    parser.add_argument("--latent_dim_s", type=int, default=512, help="shape latent dim")
    parser.add_argument("--data_path", type=str, default="../Data/IXIT1T2_s77", help="path to images")
    parser.add_argument("--out_path", type=str, default="../Tmp/",
                        help="path to output folder")
    args = parser.parse_args()
    print(args)


    diff_fac = 15
    if args.smooth:
        prefix = "GF_ADAE"
    else:
        prefix = "ADAE"

    if args.fixed_atlas:
        prefix+="_fixed"
    latent_dims = [args.latent_dim_a, args.latent_dim_s]
    save_path = args.out_path+prefix+"_"+args.loss+"_"+str(args.hospital)+"_"+args.modality+"_latent"\
                +str(latent_dims[0])+"_"+str(latent_dims[1])+"/N"+str(args.n_training)

    for fold in range(args.n_folds):
        path = save_path+"/Fold"+str(fold)
        if not os.path.exists(path):
            os.makedirs(path)

        if torch.cuda.is_available():
            device = torch.device("cuda:"+str(args.device_nr)) if torch.cuda.device_count() > 2 else torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print("Computing on: "+str(device))

        train_data = IXI_Dataset_Grayvalues(args.data_path,num_samples=args.n_training,mode="train", seq=args.modality,
                                            hospital=args.hospital, k=fold)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = 10
        np.random.seed(32)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        data_loader_train = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=train_sampler)
        data_loader_val = DataLoader(dataset=train_data, batch_size=1, sampler=valid_sampler)

        test_data = IXI_Dataset_Grayvalues(args.data_path,num_samples=args.n_training,mode="test", seq=args.modality,
                                           hospital=args.hospital, k=fold)


        data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

        if args.train:
            train(args.epochs, latent_dims, path, device, data_loader_train, data_loader_val, loss_type=args.loss,
                  smooth=args.smooth, fixed_atlas=args.fixed_atlas)
            torch.cuda.empty_cache()
        if args.test:
            test(latent_dims, path, device, data_loader_test)
        torch.cuda.empty_cache()



