import torch.optim as optim
import torch
import matplotlib.pyplot as plt


def sequence_loss(four_preds, flow_gt, H, gamma, args):
    """ Loss function defined over sequence of flow predictions """

    flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
    flow_4cor[:,:, 0, 0]  = flow_gt[:,:, 0, 0]
    flow_4cor[:,:, 0, 1] = flow_gt[:,:,  0, -1]
    flow_4cor[:,:, 1, 0] = flow_gt[:,:, -1, 0]
    flow_4cor[:,:, 1, 1] = flow_gt[:,:, -1, -1]

    ce_loss = 0.0

    for i in range(args.iters_lev0):
        i_weight = gamma**(args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()
    #
    for i in range(args.iters_lev0, args.iters_lev1 + args.iters_lev0):
        i_weight = gamma ** (args.iters_lev1 + args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    mace = torch.sum((four_preds[-1] - flow_4cor)**2, dim=1).sqrt()

    metrics = {
        '1px': (mace < 1).float().mean().item(),
        '3px': (mace < 3).float().mean().item(),
        'mace': mace.mean().item(),
    }

    return ce_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()

def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_mace_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()



