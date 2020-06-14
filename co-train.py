import argparse
import numpy as np
import torch
import utils_ct as utils
import os
from model_ct import GHNN
import pickle
import time
import datetime
from settings import settings

def train(args):
    if "ICEWS" in args.dataset:
        settings['CI'] = .5
        settings['time_scale'] = 24
        tpred_a, tpred_b, tpred_c = 1, 3, 10 # in days
        loss_scaling_factor = 1e-3
    elif "GDELT" in args.dataset:
        settings['CI'] = 1
        settings['time_scale'] = 1
        tpred_a, tpred_b, tpred_c = 60, 180, 600  # in minutes
        loss_scaling_factor = 5e-5
    else:
        raise NotImplemented

    use_cuda = torch.cuda.is_available()
    num_nodes, num_rels, num_t = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, _ = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
    test_data, _ = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
    test_filtering_file_objs = '/test_subcentric_filtering_objs.txt'
    test_filtering_file_subs = '/test_objcentric_filtering_subs.txt'

    with open('./data/' + args.dataset + test_filtering_file_objs, 'r') as f:
        _test_subcentric_filter_objs = f.readlines()
        test_subcentric_filter_objs = [[int(x.strip()) for x in line[1:-2].split(',')] for line in
                                      _test_subcentric_filter_objs]
    with open('./data/' + args.dataset + test_filtering_file_subs, 'r') as f:
        _test_objcentric_filter_subs = f.readlines()
        test_objcentric_filter_subs = [[int(x.strip()) for x in line[1:-2].split(',')] for line in
                                      _test_objcentric_filter_subs]

    train_sub_tp = '/tpre_approach/train_history_sub_tpre.txt'
    train_ob_tp = '/tpre_approach/train_history_ob_tpre.txt'
    train_sub_dt_tp = '/tpre_approach/train_history_sub_dt_tpre.txt'
    train_ob_dt_tp = '/tpre_approach/train_history_ob_dt_tpre.txt'

    test_sub_tp = '/tpre_approach/test_history_sub_tpre.txt'
    test_ob_tp = '/tpre_approach/test_history_ob_tpre.txt'
    test_sub_dt_tp = '/tpre_approach/test_history_sub_dt_tpre.txt'
    test_ob_dt_tp = '/tpre_approach/test_history_ob_dt_tpre.txt'

    train_dur =  '/tpre_approach/train_dur_last_t_tpre.txt'
    test_dur = '/tpre_approach/test_dur_last_t_tpre.txt'
    train_sub_synchro_dt = '/tpre_approach/train_sub_synchro_dt_tpre.txt'
    train_obj_synchro_dt = '/tpre_approach/train_obj_synchro_dt_tpre.txt'
    test_sub_synchro_dt = '/tpre_approach/test_sub_synchro_dt_tpre.txt'
    test_obj_synchro_dt = '/tpre_approach/test_obj_synchro_dt_tpre.txt'

    train_sub_lk = '/train_history_sub1.txt'
    train_ob_lk = '/train_history_ob1.txt'
    train_sub_dt_lk = '/train_history_sub_dt1.txt'
    train_ob_dt_lk = '/train_history_ob_dt1.txt'

    test_sub_lk = '/test_history_sub1.txt'
    test_ob_lk = '/test_history_ob1.txt'
    test_sub_dt_lk = '/test_history_sub_dt1.txt'
    test_ob_dt_lk = '/test_history_ob_dt1.txt'

    with open('./data/' + args.dataset + train_sub_tp, 'rb') as f:
        s_history_event_tp = pickle.load(f)
    with open('./data/' + args.dataset + train_ob_tp, 'rb') as f:
        o_history_event_tp = pickle.load(f)
    with open('./data/' + args.dataset + train_sub_dt_tp, 'rb') as f:
        s_history_dt_tp = pickle.load(f)
    with open('./data/' + args.dataset + train_ob_dt_tp, 'rb') as f:
        o_history_dt_tp = pickle.load(f)
    with open('./data/' + args.dataset + train_dur, 'rb') as f:
        dur_last_data = pickle.load(f)
    with open('./data/' + args.dataset + train_sub_synchro_dt, 'rb') as f:
        sub_synchro_dt_data = pickle.load(f)
    with open('./data/' + args.dataset + train_obj_synchro_dt, 'rb') as f:
        obj_synchro_dt_data = pickle.load(f)
    with open('./data/' + args.dataset + train_sub_lk, 'rb') as f:
        s_history_event_lk = pickle.load(f)
    with open('./data/' + args.dataset + train_ob_lk, 'rb') as f:
        o_history_event_lk = pickle.load(f)
    with open('./data/' + args.dataset + train_sub_dt_lk, 'rb') as f:
        s_history_dt_lk = pickle.load(f)
    with open('./data/' + args.dataset + train_ob_dt_lk, 'rb') as f:
        o_history_dt_lk = pickle.load(f)

    with open('./data/' + args.dataset + test_sub_tp, 'rb') as f:
        s_history_test_event_tp = pickle.load(f)
    with open('./data/' + args.dataset + test_ob_tp, 'rb') as f:
        o_history_test_event_tp = pickle.load(f)
    with open('./data/' + args.dataset + test_sub_dt_tp, 'rb') as f:
        s_history_test_dt_tp = pickle.load(f)
    with open('./data/' + args.dataset + test_ob_dt_tp, 'rb') as f:
        o_history_test_dt_tp = pickle.load(f)
    with open('./data/' + args.dataset + test_dur, 'rb') as f:
        dur_last_test = pickle.load(f)
    with open('./data/' + args.dataset + test_sub_synchro_dt, 'rb') as f:
        sub_synchro_dt_test = pickle.load(f)
    with open('./data/' + args.dataset + test_obj_synchro_dt, 'rb') as f:
        obj_synchro_dt_test = pickle.load(f)
    with open('./data/' + args.dataset + test_sub_lk, 'rb') as f:
        s_history_test_event_lk = pickle.load(f)
    with open('./data/' + args.dataset + test_ob_lk, 'rb') as f:
        o_history_test_event_lk = pickle.load(f)
    with open('./data/' + args.dataset + test_sub_dt_lk, 'rb') as f:
        s_history_test_dt_lk = pickle.load(f)
    with open('./data/' + args.dataset + test_ob_dt_lk, 'rb') as f:
        o_history_test_dt_lk = pickle.load(f)

    model = GHNN(num_nodes, num_rels, num_t, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if use_cuda:
        model.cuda()

    if not args.only_eva:
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y,%H:%M:%S") + args.timepre_mode + args.dataset
        main_dirName = os.path.join(args.save_dir, dt_string)
        if not os.path.exists(main_dirName):
            os.makedirs(main_dirName)

        model_path = os.path.join(main_dirName, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        settings['main_dirName'] = main_dirName
        file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
        file_training.write("Training Configuration: \n")
        for key in settings:
            file_training.write(key + ': ' + str(settings[key]) + '\n')
        for arg in vars(args):
            file_training.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        print("start training...")
        file_training.write("Training Start \n")
        file_training.write("===============================\n")

        epoch = 0
        while True:
            model.train()
            if epoch == args.max_epochs:
                break
            epoch += 1
            loss_epoch = 0
            t0 = time.time()
            _batch = 0
            for batch_data in utils.make_batch(train_data, s_history_event_tp,
                    s_history_event_lk, o_history_event_tp, o_history_event_lk, s_history_dt_tp, s_history_dt_lk, o_history_dt_tp,
                    o_history_dt_lk, dur_last_data, sub_synchro_dt_data, obj_synchro_dt_data, args.batch_size):
                batch_data[0] = torch.from_numpy(batch_data[0])
                if use_cuda:
                    batch_data[0] = batch_data[0].cuda()
                cro_entr_lk, error_tp, _, _, _, _, _, _, _,_ = model(batch_data, args.timepre_mode, 'Training')
                if cro_entr_lk is not None and error_tp is not None:
                    error = cro_entr_lk + loss_scaling_factor * error_tp
                elif cro_entr_lk is not None:
                    error = cro_entr_lk
                elif error_tp is not None:
                    error = args.error_tp * error_tp
                else:
                    continue
                error.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += error.item()

                print("batch: " + str(_batch) + ' finished. Accumulated used time: ' + str(time.time() - t0)+
                    ', Error: '+ str(error) + '\n')
                file_training.write(
                    "epoch: " + str(epoch) + "batch: " + str(_batch) + ' finished. Accumulated used time: '
                    + str(time.time() - t0)+  ', Error: '+ str(error) + '\n')

                _batch += 1
            t3 = time.time()
            print("Epoch {:04d} | Error {:.4f}| time {:.4f}".
                  format(epoch, loss_epoch / _batch, t3 - t0))
            file_training.write("Epoch {:04d} | Error {:.4f}| time {:.4f}".
                  format(epoch, loss_epoch /_batch, t3 - t0) + '\n')

        torch.save(model, model_path + '/' + args.dataset + '_best.pth')
        print("training done")
        file_training.write("training done")
        file_training.close()


    ##### Evaluation
    if args.only_eva:
        main_dirName = args.eva_dir
        model_path = os.path.join(main_dirName, 'models')
    if args.filtering:
        file_test_path = os.path.join(main_dirName, "test_record_filtering.txt")
    else:
        file_test_path = os.path.join(main_dirName, "test_record_raw.txt")
    file_test = open(file_test_path, "w")
    file_test.write("Testing starts: \n")
    model = torch.load(model_path + '/' + args.dataset + '_best.pth')
    model.eval()
    mae_tp_epoch = 0
    _batch = 0
    abs_error_tp = []
    subcent_ranks_lk = []
    obcent_ranks_lk = []
    for batch_data in utils.make_batch(test_data, s_history_test_event_tp, s_history_test_event_lk,
                                       o_history_test_event_tp, o_history_test_event_lk, s_history_test_dt_tp,
                                       s_history_test_dt_lk,
                                       o_history_test_dt_tp, o_history_test_dt_lk, dur_last_test,
                                       sub_synchro_dt_test,
                                       obj_synchro_dt_test, args.batch_size, valid1=test_subcentric_filter_objs,
                                       valid2=test_objcentric_filter_subs):
        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0]))
        _batch += 1
        with torch.no_grad():
            sub_rank, obj_rank, cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, gt_tp, den1_tp, den2_tp, \
            tpred, abs_error = model(batch_data, args.timepre_mode, 'Test')
            mae_tp_epoch += mae_tp.item()
            abs_error_tp.append(abs_error.squeeze())
            subcent_ranks_lk += sub_rank
            obcent_ranks_lk += obj_rank

    # compute Hits@k of time prediction
    abs_error_tp = torch.cat(abs_error_tp, dim=0)
    for hit in [tpred_a, tpred_b, tpred_c]:
        avg_count_tp = (abs_error_tp <= hit).type(torch.float).mean()
        file_test.write("Test Hits (tp) @ {}: {:.6f}".format(hit, avg_count_tp) + '\n')

    # evaluation matrices of link prediction
    subcent_ranks_lk = np.asarray(subcent_ranks_lk)
    obcent_ranks_lk = np.asarray(obcent_ranks_lk)
    subcent_mr_lk = np.mean(subcent_ranks_lk)
    objcent_mr_lk = np.mean(obcent_ranks_lk)
    subcent_mrr_lk = np.mean(1.0 / subcent_ranks_lk)
    objcent_mrr_lk = np.mean(1.0 / obcent_ranks_lk)
    total_ranks_lk = np.concatenate((subcent_ranks_lk, obcent_ranks_lk))
    mr_lk = np.mean(total_ranks_lk)
    mrr_lk = np.mean(1.0 / total_ranks_lk)

    file_test.write("sub centic test MRR (lk): {:.6f}".format(subcent_mrr_lk) + '\n')
    file_test.write("obj centic test MRR (lk): {:.6f}".format(objcent_mrr_lk) + '\n')
    file_test.write("total test MRR (lk): {:.6f}".format(mrr_lk) + '\n')
    file_test.write("sub centric test MR (lk): {:.6f}".format(subcent_mr_lk) + '\n')
    file_test.write("obj centric test MR (lk): {:.6f}".format(objcent_mr_lk) + '\n')
    file_test.write("total test MR (lk): {:.6f}".format(mr_lk) + '\n')
    if "GDELT" in args.dataset:
        mae_tp_epoch = mae_tp_epoch / 60.0 #converting the value from minutes to hours
    file_test.write("Test MAE: " + str(mae_tp_epoch / _batch) + '\n')

    for hit in [1, 3, 10]:
        avg_count_lk = np.mean((total_ranks_lk <= hit))
        avg_count_sub_lk = np.mean((subcent_ranks_lk <= hit))
        avg_count_obj_lk = np.mean((obcent_ranks_lk <= hit))

        file_test.write("Test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_lk) + '\n')
        file_test.write("sub centric test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')
        file_test.write("obj centric test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GHNN')
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--max_hist_len", type=int, default=10)  # maximum history sequence length
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved")
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--timepre_mode", type=str, default='MSE')
    parser.add_argument("-d", "--dataset", type=str, default='GDELT', help="dataset to use")
    parser.add_argument("--embd_rank", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=10, help="maximum epochs")
    parser.add_argument("--softrelu_scale", type=float, default=10.0)
    parser.add_argument("--filtering", type= utils.str2bool, default=True)
    parser.add_argument("--only_eva", type=utils.str2bool, default=False, help="whether only evaluation on test set")
    parser.add_argument("--eva_dir", type = str, default= "saved/08-06-2020,19:05:51MSEICEWS14", help="saved dir of the testing model")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)


